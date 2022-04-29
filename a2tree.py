"""
Assignment 2: Quadtree Compression

=== CSC148 Winter 2021 ===
Department of Mathematical and Computational Sciences,
University of Toronto Mississauga

=== Module Description ===
This module contains classes implementing the quadtree.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from copy import deepcopy


# No other imports allowed


def is_empty(x) -> bool:
    """
    Return whether the List is empty recursively
    """
    for item in x:
        if not isinstance(item, list) or not is_empty(item):
            return False
    return True


def is_even(num: int) -> bool:
    """
    Return whether the int is even
    """
    return num % 2 == 0


def mean_and_count(matrix: List[List[int]]) -> Tuple[float, int]:
    """
    Returns the average of the values in a 2D list
    Also returns the number of values in the list
    """
    total = 0
    count = 0
    for row in matrix:
        for v in row:
            total += v
            count += 1
    return total / count, count


def standard_deviation_and_mean(matrix: List[List[int]]) -> Tuple[float, float]:
    """
    Return the standard deviation and mean of the values in <matrix>

    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Note that the returned average is a float.
    It may need to be rounded to int when used.
    """
    avg, count = mean_and_count(matrix)
    total_square_error = 0
    for row in matrix:
        for v in row:
            total_square_error += ((v - avg) ** 2)
    return math.sqrt(total_square_error / count), avg


def merge(bl, br, tl, tr, height):
    final_lst = []

    h = height//2

    for i in range(h):
        final_lst.append(bl[i] + br[i])

    for i in range(height - h):
        final_lst.append(tl[i] + tr[i])

    return final_lst


def losses(og, cm):
    max_loss = 0
    if isinstance(cm, QuadTreeNodeLeaf):
        if isinstance(og, QuadTreeNodeInternal):
            pix = super_flat(og.trav_tree())
            return st_d(pix)
        else:
            return 0
    if isinstance(og, QuadTreeNodeInternal) and \
            isinstance(cm, QuadTreeNodeInternal):
        for i in range(4):
            max_loss = max(max_loss, losses(og.children[i], cm.children[i]))
    return max_loss


def mean(pixels: List[int]) -> int:
    return round(sum(pixels) / len(pixels))


def st_d(pixels: List[int]) -> float:
    count = len(pixels)
    m = mean(pixels)
    arr = [(x - m)**2 for x in pixels]
    std = math.sqrt(sum(arr)/count)
    return std


def super_flat(obj):
    t = []
    if isinstance(obj, int):
        t.append(obj)
    else:
        for sub in obj:
            t.extend(super_flat(sub))
    return t


class QuadTreeNode:
    """
    Base class for a node in a quad tree
    """

    def __init__(self) -> None:
        pass

    def tree_size(self) -> int:
        raise NotImplementedError

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        raise NotImplementedError

    def preorder(self) -> str:
        raise NotImplementedError


class QuadTreeNodeEmpty(QuadTreeNode):
    """
    An empty node represents an area with no pixels included
    """

    def __init__(self) -> None:
        super().__init__()

    def tree_size(self) -> int:
        """
        Note: An empty node still counts as 1 node in the quad tree
        """
        return 1

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Convert to a properly formatted empty list
        """
        # Note: Normally, this method should return an empty list or a list of
        # empty lists. However, when the tree is mirrored, this returned list
        # might not be empty and may contain the value 255 in it. This will
        # cause the decompressed image to have unexpected white pixels.
        # You may ignore this caveat for the purpose of this assignment.
        return [[255] * width for _ in range(height)]

    def preorder(self) -> str:
        """
        The letter E represents an empty node
        """
        return 'E'

    def copy(self):
        return QuadTreeNodeEmpty()

    def pix_swap(self):
        pass


class QuadTreeNodeLeaf(QuadTreeNode):
    """
    A leaf node in the quad tree could be a single pixel or an area in which
    all pixels have the same colour (indicated by self.value).
    """

    value: int  # the colour value of the node

    def __init__(self, value: int) -> None:
        super().__init__()
        assert isinstance(value, int)
        self.value = value

    def tree_size(self) -> int:
        """
        Return the size of the subtree rooted at this node
        """
        return 1

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Return the pixels represented by this node as a 2D list

        >>> sample_leaf = QuadTreeNodeLeaf(5)
        >>> sample_leaf.convert_to_pixels(2, 2)
        [[5, 5], [5, 5]]
        """
        pix = []
        for i in range(height):
            row = []
            for j in range(width):
                row.append(self.value)
            pix.append(row)

        return pix

    def copy(self):
        return QuadTreeNodeLeaf(self.value)

    def preorder(self) -> str:
        """
        A leaf node is represented by an integer value in the preorder string
        """
        return str(self.value)

    def pix_swap(self):
        pass


class QuadTreeNodeInternal(QuadTreeNode):
    """
    An internal node is a non-leaf node, which represents an area that will be
    further divided into quadrants (self.children).

    The four quadrants must be ordered in the following way in self.children:
    bottom-left, bottom-right, top-left, top-right

    (List indices increase from left to right, bottom to top)

    Representation Invariant:
    - len(self.children) == 4
    """
    children: List[Optional[QuadTreeNode]]

    def __init__(self) -> None:
        """
        Order of children: bottom-left, bottom-right, top-left, top-right
        """
        super().__init__()

        # Length of self.children must be always 4.
        self.children = [None, None, None, None]

    def tree_size(self) -> int:
        """
        The size of the subtree rooted at this node.

        This method returns the number of nodes that are in this subtree,
        including the root node.
        """
        count = 1
        for child in self.children:
            count += child.tree_size()

        return count

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Return the pixels represented by this node as a 2D list.

        You'll need to recursively get the pixels for the quadrants and
        combine them together.

        Make sure you get the sizes (width/height) of the quadrants correct!
        Read the docstring for split_quadrants() for more info.
        """
        bl, br, tl, tr = self.children

        w = width // 2
        h = height // 2

        b_l, b_r, t_l, t_r = (bl.convert_to_pixels(w, h),
                              br.convert_to_pixels(width - w, h),
                              tl.convert_to_pixels(w, height - h),
                              tr.convert_to_pixels(width - w, height - h))

        return merge(b_l, b_r, t_l, t_r, height)

    def preorder(self) -> str:
        """
        Return a string representing the preorder traversal or the tree rooted
        at this node. See the docstring of the preorder() method in the
        QuadTree class for more details.

        An internal node is represented by an empty string in the preorder
        string.
        """
        order_str = ''
        for child in self.children:
            order_str += ','
            if child is not None:
                order_str += child.preorder()

        return order_str

    def restore_from_preorder(self, lst: List[str], start: int) -> int:
        """
        Restore subtree from preorder list <lst>, starting at index <start>
        Return the number of entries used in the list to restore this subtree
        """

        # This assert will help you find errors.
        # Since this is an internal node, the first entry to restore should
        # be an empty string
        assert lst[start] == ''

        index, total, self.children = start + 1, len(lst), []

        while index < total and len(self.children) < 4:
            leaf = lst[index]

            if leaf == 'E':
                kid = QuadTreeNodeEmpty()
            elif leaf == '':
                kid = QuadTreeNodeInternal()
                index = - 1 + kid.restore_from_preorder(lst, index)
            else:
                kid = QuadTreeNodeLeaf(int(leaf))
            index += 1
            self.children.append(kid)

        return index

    def copy(self):
        x = QuadTreeNodeInternal()
        for i in range(4):
            x.children[i] = self.children[i].copy()

        return x

    def pix_swap(self):
        self.children[0], self.children[2] = self.children[2], self.children[0]
        self.children[1], self.children[3] = self.children[3], self.children[1]
        for i in range(4):
            node = self.children[i]
            if isinstance(node, QuadTreeNodeInternal):
                node.pix_swap()

    def mirror(self) -> None:
        """
        Mirror the bottom half of the image represented by this tree over
        the top half

        Example:
            Original Image
            1 2
            3 4

            Mirrored Image
            3 4 (this row is flipped upside down)
            3 4

        See the assignment handout for a visual example.
        """

        bl_copy, br_copy = self.children[0].copy(), self.children[1].copy()

        bl_copy.pix_swap()
        br_copy.pix_swap()

        self.children[2] = bl_copy
        self.children[3] = br_copy

    def trav_tree(self):
        pix = []
        for child in self.children:
            if isinstance(child, QuadTreeNodeLeaf):
                pix.append([child.value])
            elif isinstance(child, QuadTreeNodeInternal):
                pix.append(child.trav_tree())
        return pix


class QuadTree:
    """
    The class for the overall quadtree
    """

    loss_level: float
    height: int
    width: int
    root: Optional[QuadTreeNode]  # safe to assume root is an internal node

    def __init__(self, loss_level: int = 0) -> None:
        """
        Precondition: the size of <pixels> is at least 1x1
        """
        self.loss_level = float(loss_level)
        self.height = -1
        self.width = -1
        self.root = None

    def __eq__(self, other):
        return self.root == other.root

    def build_quad_tree(self, pixels: List[List[int]],
                        mirror: bool = False) -> None:
        """
        Build a quad tree representing all pixels in <pixels>
        and assign its root to self.root

        <mirror> indicates whether the compressed image should be mirrored.
        See the assignment handout for examples of how mirroring works.
        """
        # print('building_quad_tree...')
        self.height = len(pixels)
        self.width = len(pixels[0])
        self.root = self._build_tree_helper(pixels)
        if mirror:
            self.root.mirror()
        return

    def _build_tree_helper(self, pixels: List[List[int]]) -> QuadTreeNode:
        """
        Build a quad tree representing all pixels in <pixels>
        and return the root

        Note that self.loss_level should affect the building of the tree.
        This method is where the compression happens.

        IMPORTANT: the condition for compressing a quadrant is the standard
        deviation being __LESS THAN OR EQUAL TO__ the loss level. You must
        implement this condition exactly; otherwise, you could fail some
        test cases unexpectedly.
        """
        if is_empty(pixels):
            return QuadTreeNodeEmpty()
        else:
            std, mean = standard_deviation_and_mean(pixels)
            if std <= self.loss_level:
                return QuadTreeNodeLeaf(round(mean))
            elif len(pixels) == 1 and len(pixels[0]) == 1:
                return QuadTreeNodeLeaf(pixels[0][0])
            else:
                root = QuadTreeNodeInternal()
                split = self._split_quadrants(pixels)

                root.children = [self._build_tree_helper(split[0]),
                                 self._build_tree_helper(split[1]),
                                 self._build_tree_helper(split[2]),
                                 self._build_tree_helper(split[3])]

        return root

    @staticmethod
    def _split_quadrants(pixels: List[List[int]]) -> List[List[List[int]]]:
        """
        Precondition: size of <pixels> is at least 1x1
        Returns a list of four lists of lists, correspoding to the quadrants in
        the following order: bottom-left, bottom-right, top-left, top-right

        IMPORTANT: when dividing an odd number of entries, the smaller half
        must be the left half or the bottom half, i.e., the half with lower
        indices.

        Postcondition: the size of the returned list must be 4

        >>> example = QuadTree(0)
        >>> example._split_quadrants([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
        [[[1]], [[2, 3]], [[4], [7]], [[5, 6], [8, 9]]]
        """
        height = len(pixels)
        width = len(pixels[0])

        b_l = []
        b_r = []
        t_l = []
        t_r = []

        h_mid = height // 2
        w_mid = width // 2

        bottom = pixels[:h_mid]
        top = pixels[h_mid:]
        for row in bottom:
            b_l.append(row[:w_mid])
            b_r.append(row[w_mid:])
        for row in top:
            t_l.append(row[:w_mid])
            t_r.append(row[w_mid:])

        return [b_l, b_r, t_l, t_r]

    def tree_size(self) -> int:
        """
        Return the number of nodes in the tree, including all Empty, Leaf, and
        Internal nodes.
        """
        return self.root.tree_size()

    def convert_to_pixels(self) -> List[List[int]]:
        """
        Return the pixels represented by this tree as a 2D matrix
        """
        return self.root.convert_to_pixels(self.width, self.height)

    def preorder(self) -> str:
        """
        return a string representing the preorder traversal of the quadtree.
        The string is a series of entries separated by comma (,).
        Each entry could be one of the following:
        - empty string '': represents a QuadTreeNodeInternal
        - string of an integer value such as '5': represents a QuadTreeNodeLeaf
        - string 'E': represents a QuadTreeNodeEmpty

        For example, consider the following tree with a root and its 4 children
                __      Root       __
              /      |       |        \
            Empty  Leaf(5), Leaf(8), Empty

        preorder() of this tree should return exactly this string: ",E,5,8,E"

        (Note the empty-string entry before the first comma)
        """
        return self.root.preorder()

    @staticmethod
    def restore_from_preorder(lst: List[str],
                              width: int, height: int) -> QuadTree:
        """
        Restore the quad tree from the preorder list <lst>
        The preorder list <lst> is the preorder string split by comma

        Precondition: the root of the tree must be an internal node (non-leaf)
        """
        tree = QuadTree()
        tree.width = width
        tree.height = height
        tree.root = QuadTreeNodeInternal()
        tree.root.restore_from_preorder(lst, 0)
        return tree


def maximum_loss(original: QuadTreeNode, compressed: QuadTreeNode) -> float:
    """
    Given an uncompressed image as a quad tree and the compressed version,
    return the maximum loss across all compressed quadrants.

    Precondition: original.tree_size() >= compressed.tree_size()

    Note: original, compressed are the root nodes (QuadTreeNode) of the
    trees, *not* QuadTree objects

    >>> pixels = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> orig, comp = QuadTree(0), QuadTree(2)
    >>> orig.build_quad_tree(pixels)
    >>> comp.build_quad_tree(pixels)
    >>> maximum_loss(orig.root, comp.root)
    1.5811388300841898
    """
    return losses(original, compressed)


if __name__ == '__main__':

    import doctest
    doctest.testmod()

    p1 = [[1, 2, 3], [4, 5, 6]]

    p2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    p3 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    p4 = [[1, 2], [3, 4]]

    p5 = [[1], [2], [3], [4], [5]]

    p6 = [[1, 2, 3]]

    p7 = [[100, 150, 200], [70, 35, 21], [43, 230, 109]]

    p8 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    p9 = [[13, 14, 15, 16], [13, 14, 15, 16], [1, 2, 3, 4],
          [70, 35, 21, 54], [9, 10, 11, 12], [13, 14, 15, 16],
          [1, 2, 3, 4], [13, 14, 15, 16], [5, 6, 7, 8]]

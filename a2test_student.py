"""
Assignment 2: Quadtree Compression

=== CSC148 Winter 2021 ===
Department of Mathematical and Computational Sciences,
University of Toronto Mississauga

=== Module Description ===
This module contains the test suite
"""

import pytest
from a2tree import QuadTree, QuadTreeNode, QuadTreeNodeEmpty, QuadTreeNodeLeaf,\
    QuadTreeNodeInternal, maximum_loss, super_flat


"""
Test cases
"""

p1 = [[1, 2, 3], [4, 5, 6]]
p2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
p3 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
p4 = [[1, 2], [3, 4]]
p5 = [[1], [2], [3], [4], [5]]
p6 = [[1, 2, 3]]
p7 = [[100, 150, 200], [70, 35, 21], [43, 230, 109]]


def test_copy_1():
    x = QuadTree()
    x.build_quad_tree(p4)
    y = x.root.copy()
    assert x.preorder() == y.preorder()


def test_trav_tree_contents():
    x = QuadTree()
    x.build_quad_tree(p4)
    y = super_flat(x.root.trav_tree())
    z = super_flat(p4)
    assert y == z


def test_copy_2():
    x = QuadTree()
    x.build_quad_tree(p2)
    y = x.root.copy()
    assert x.preorder() == y.preorder()


def test_convert_to_pixels_1():
    x = QuadTree()
    x.build_quad_tree(p1)
    assert str(p1) == str(x.convert_to_pixels())


def test_convert_to_pixels_2():
    x = QuadTree()
    x.build_quad_tree(p5)
    assert str(p5) == str(x.convert_to_pixels())


def test_convert_to_pixels_3():
    x = QuadTree()
    x.build_quad_tree(p6)
    assert str(p6) == str(x.convert_to_pixels())


def test_mirror_1():
    x = QuadTree(0)
    x.build_quad_tree(p4)
    x.root.mirror()
    assert x.convert_to_pixels() == [[1, 2], [1, 2]]


def test_mirror_2():
    x = QuadTree(0)
    x.build_quad_tree(p2)
    x.root.mirror()
    assert x.convert_to_pixels() == [[1, 2, 3], [1, 2, 3], [1, 255, 255]]


def test_mirror_3():
    x = QuadTree(0)
    x.build_quad_tree(p2, True)
    tree = x.preorder()
    assert tree == ",1,,E,E,2,3,1,,2,3,E,E"


def test_split_quadrants_1():
    x = QuadTree._split_quadrants(p1)
    assert x == [[[1]], [[2, 3]], [[4]], [[5, 6]]]


def test_split_quadrants_2():
    x = QuadTree._split_quadrants(p5)
    assert x == [[[], []], [[1], [2]], [[], [], []], [[3], [4], [5]]]


def test_split_quadrants_3():
    x = QuadTree._split_quadrants([[1]])
    assert x == [[], [], [[]], [[1]]]


def test_restore_from_preorder_1():
    x = QuadTree(0)
    x.build_quad_tree(p3)
    lst = x.preorder().split(',')

    root = QuadTreeNodeInternal()
    root.restore_from_preorder(lst, 0)

    pre1 = x.preorder()
    pre2 = root.preorder()

    assert pre1 == pre2


def test_restore_from_preorder_2():
    x = QuadTree(0)
    x.build_quad_tree(p7)
    lst = x.preorder().split(',')

    root = QuadTreeNodeInternal()
    root.restore_from_preorder(lst, 0)

    pre1 = x.preorder()
    pre2 = root.preorder()

    assert pre1 == pre2


def test_restore_from_preorder_3():
    x = QuadTree(3)
    x.build_quad_tree(p3)
    lst = x.preorder().split(',')

    root = QuadTreeNodeInternal()
    root.restore_from_preorder(lst, 0)

    pre1 = x.preorder()
    pre2 = root.preorder()

    assert pre1 == pre2


def test_max_loss_1():
    x = QuadTree(0)
    x.build_quad_tree(p3)

    y = QuadTree(0)
    y.build_quad_tree(p3)

    assert maximum_loss(x.root, y.root) == 0


def test_max_loss_2():
    x = QuadTree(0)
    x.build_quad_tree(p3)

    y = QuadTree(5)
    y.build_quad_tree(p3)

    assert maximum_loss(x.root, y.root) == 4.636809247747852


if __name__ == '__main__':

    pytest.main(['a2test_student.py'])

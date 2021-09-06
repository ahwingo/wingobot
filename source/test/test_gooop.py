"""
Unit tests for the Goban class and its helper functions.
"""

# Python Standard
from inspect import ismethod, getmembers
import sys
import unittest

# Third Party
import numpy as np

# Local
sys.path.append("..")
from source.gooop import Goban


class TestGoban(unittest.TestCase):
    """
    Test the methods of the Goban class.
    """
    def test_constructor(self):
        """
        Construct and get the size of a Goban.
        """
        g = Goban(13)
        g_total_bytes = sys.getsizeof(g)
        print("Total size of one Goban: {} MB.".format(g_total_bytes / (1024 ** 2)))
        self.assertEqual(g.size, 13)

    def test_copy(self):
        """
        Ensure that when a Goban is copied, all of its complex members are copied and not simply just referenced.
        """
        gorig = Goban(13)
        gcopy = gorig.copy()
        complex_classes = [type(np.ndarray(1))]
        get_complex_vals = lambda g: [value for key, value in getmembers(g, lambda x: not ismethod(x)) if type(value) in complex_classes]
        gorig_members = get_complex_vals(gorig)
        gcopy_members = get_complex_vals(gcopy)
        for orig_attribute, copy_attribute in zip(gorig_members, gcopy_members):
            self.assertFalse(orig_attribute is copy_attribute)

    def test_stone_conflict_illegal_move(self):
        """
        Ensure that an illegal move is not allowed.
        This move is illegal because another stone is already on that position.
        """
        g = Goban(13)
        valid = g.make_move(0, 0)
        self.assertTrue(valid)
        valid = g.make_move(0, 0)
        self.assertFalse(valid)

    def test_self_capture_illegal_move(self):
        """
        Ensure that an illegal move is not allowed.
        This move is illegal because another stone is already on that position.
        """
        g = Goban(13)
        g.make_move(0, 1)  # Black forms eye.
        g.make_move(13, 13)  # White pass.
        g.make_move(1, 0)  # Black forms eye.
        g.make_move(13, 13)  # White pass.
        g.make_move(1, 1)  # Black forms eye.
        valid = g.make_move(0, 0)  # White attempts an illegal self-capture.
        self.assertFalse(valid)

    def test_legal_move(self):
        """
        Ensure that legal moves return a success value of true.
        """
        g = Goban(13)
        valid = g.make_move(0, 0)
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
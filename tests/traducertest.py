import sys #needed to import modules from parent directories

sys.path.append('../code')

import numpy as np
import os
import ROOT as root
import unittest
from roottonumpy import RootToNumpy
from dummytree import MakeDummyTree



class TestTraducer(unittest.TestCase):

    def test_traduction(self):

        elem_num = 100

        MakeDummyTree(elem_num, "dummytree.root")

        translated_tree = RootToNumpy( "dummytree.root", "tree") 

        for i in range(elem_num):
            for j in range(20):
                self.assertAlmostEqual(translated_tree[i][j], (i*20 + j))


if __name__ == '__main__':
    unittest.main()








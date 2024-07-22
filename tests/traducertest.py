import sys
# needed to import modules from parent directories
sys.path.append('../code')
import numpy as np
import os
import ROOT as root
import unittest
from traducers import RootToNumpy, NumpyToRoot, ArrayDimensionError
from dummytree import MakeDummyTree


class TestTraducer(unittest.TestCase):

    def test_RootToNumpy(self):

        elem_num = 100 # number of elements in the dummy tree

        MakeDummyTree(elem_num, "dummytree.root")

        translated_tree = RootToNumpy("dummytree.root", "tree") 

        #checking if dummytree has been written correctly
        
        for i in range(elem_num):
            for j in range(20):
                self.assertAlmostEqual(translated_tree[i][j], (i*20 + j))

    def test_NumpyToRoot(self):

        elem_num = 100 # number of elements in the dummy array
        
        a = np.zeros((elem_num, 20), dtype=np.double)  # initializing empty dummy array to be filled

        for i in range(elem_num):  # filling dummy array with consecutive numbers
            for j in range(20):
                a[i][j] = i*20 + j 

        NumpyToRoot("a.root", "tree", a)  # translating a into a .root file

        a_retraduced = RootToNumpy("a.root", "tree")  # bringing a back to test that nothing wrong happened

        for i in range(elem_num):
            for j in range(20):
                self.assertAlmostEqual(a_retraduced[i][j], (i*20 + j))

        with self.assertRaises(TypeError):  # check that raises TypeError with incorrect inputs
            b = "pippo"
            NumpyToRoot("a.root", "tree", b)

        # checking that raises exception when given array of the wrong shape

        with self.assertRaises(ArrayDimensionError):
            c = np.zeros(20, dtype=np.double) 
            NumpyToRoot("a.root", "tree", c)
        
        with self.assertRaises(ArrayDimensionError):
            c = np.zeros((5, 3, 7), dtype=np.double) 
            NumpyToRoot("a.root", "tree", c)


if __name__ == '__main__':
    unittest.main()








import numpy as np
import os
from array import array
import ROOT as root

def MakeDummyTree(M, a, b): #makes a tree with M events, two columns, one with vectors of a elements and the other with vectors of b elements

    name = "dummytree.root"

    path = "./" + name

    if os.path.exists(path):

        os.remove(path)

    file = root.TFile.Open(name, "RECREATE")

    tree = root.TTree("tree", "tree")

    paramvector = array('d', a*[0.]) 
    covariancematrix = array('d', b*[0.])

    tree.Branch("Vector", paramvector, 'Vector/D')
    tree.Branch("Matrix", covariancematrix, 'Matrix/D')

    for i in range(M):

        for j in range(a):

            paramvector[j] =  i*(a+b) + j

        for k in range(b):

            covariancematrix[k] = i*(a+b) + a + k

        tree.Fill()

    file.Write()

if __name__ == '__main__':

    unittest.main()
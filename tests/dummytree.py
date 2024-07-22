import numpy as np
import os
import ROOT as root

def MakeDummyTree(M, name = "dummytree.root"): #makes a tree with M events, two columns, one with vectors of a elements and the other with vectors of b elements, filled with consecutive numbers

    path = "./" + name

    if os.path.exists(path):

        os.remove(path) #if the file containing the dummy tree already exists, it is canceled

    file = root.TFile.Open(name, "RECREATE")

    tree = root.TTree("tree", "tree")

    paramvector = np.zeros((5,), dtype=np.double)
    covariancematrix = np.zeros((15,), dtype=np.double)

    tree.Branch("Vector", paramvector, 'Vector[5]/D')
    tree.Branch("Matrix", covariancematrix, 'Matrix[15]/D')

    for i in range(M):

        for j in range(5):

            paramvector[j] =  i*20 + j

        for k in range(15):

            covariancematrix[k] = i*20 + 5 + k

        tree.Fill()

    file.Write()

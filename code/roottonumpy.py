import numpy as np
import ROOT as root

def RootToNumpy(name, treename):

    file = root.TFile.Open(name, "READ")
    tree = file.Get(treename)
    a = [] #initializing an empty list

    for entry in tree:
        x1 = np.array(entry.Vector) #getting values of vector branch
        x2 = np.array(entry.Matrix) #getting values of matrix branch
        x = np.concatenate((x1, x2), axis=None) #fusing into a single vector
        a.append(x)

    a = np.array(a) #converting a into a numpy array

    return a




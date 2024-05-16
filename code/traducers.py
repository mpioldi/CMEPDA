import numpy as np
import os
import ROOT as root

class ArrayDimensionError(BaseException):
    '''
    This exception catches when the input arrays in the traducer from 
    numpy to root are not of the correct length, that is 20
    '''

    def __init__(self):
        super().__init__('Input array has not the correct dimension')


def RootToNumpy(name, treename):

    try:
        file = root.TFile.Open(name, "READ")
        tree = file.Get(treename)

    except OSError as e:
        print(f'Impossible to read the file. \n{e}') #message shown if file or tree is not found
    
    else:

        a = [] #initializing an empty list

        for entry in tree:
            x1 = np.array(tree.Vector) #getting values of vector branch
            x2 = np.array(tree.Matrix) #getting values of matrix branch
            x = np.concatenate((x1, x2), axis=None) #fusing into a single vector
            a.append(x)

        a = np.array(a) #converting a into a numpy array

        return a
        
def RootTotxt(name, treename):

    try:
        file = root.TFile.Open(name, "READ")
        tree = file.Get(treename)

    except OSError as e:
        print(f'Impossible to read the file. \n{e}') #message shown if file or tree is not found
    
    else:

        a = [] #initializing an empty list

        for entry in tree:
            x1 = np.array(tree.Vector) #getting values of vector branch
            x2 = np.array(tree.Matrix) #getting values of matrix branch
            x = np.concatenate((x1, x2), axis=None) #fusing into a single vector
            a.append(x)

        a = np.array(a) #converting a into a numpy array
        
        
        outputname = name.removesuffix('.root') + '.txt' #name of the output
        
        head = 'qoverp lambda phi dxy dsz cov_1 cov_2 cov_3 cov_4 cov_5 cov_6 cov_7 cov_8 cov_9 cov_10 cov_11 cov_12 cov_13 cov_14 cov_15' #header of the file
        
        np.savetxt(outputname, a, delimiter=' ', newline='\n', header=head)

        return
        
        
def NumpyToRoot(name, treename, myarray):

    num_elem = len(myarray)

    if not isinstance(myarray, np.ndarray):
        raise TypeError('Input is not a numpy array') #Tells when input is not a numpy array
    if myarray.shape != (num_elem, 20):
        raise ArrayDimensionError #Tells when input has not the correct dimension

    path = "./" + name

    if os.path.exists(path):

        os.remove(path) #if the tree already exists, it is canceled

    file = root.TFile.Open(name, "RECREATE")

    tree = root.TTree("tree", "tree")

    paramvector = np.zeros((5,), dtype=np.double)
    covariancematrix = np.zeros((15,), dtype=np.double)

    '''
    tree branches made of arrays of length 5 and 15 containing the parameters
    and the covariance matrix are created
    '''

    tree.Branch("Vector", paramvector, 'Vector[5]/D') 
    tree.Branch("Matrix", covariancematrix, 'Matrix[15]/D')

    for i in range(num_elem):

        for j in range(5):

            paramvector[j] =  myarray[i][j]

        for k in range(15):

            covariancematrix[k] = myarray[i][5 + k]

        tree.Fill()

    file.Write()




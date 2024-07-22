import numpy as np
import matplotlib.pyplot as plt

#import initial data

initial = np.loadtxt("data.txt")[::500]

#import final data

final = np.loadtxt("decompr_data.txt")

#initialize dummy matrices

initial_matrix = np.zeros((5, 5))

final_matrix = np.zeros((5, 5))

#initialize eignevector matrices

initial_eigenvs = np.zeros((len(initial), 5))

final_eigenvs = np.zeros((len(final), 5))

#convert each row into matrix and take its eigenvalues

for i in range(len(initial)):
    
    for j in range(5):

        for k in range(j+1):

            initial_matrix[k,j] = initial[i,j*(j+1)//2 + k +4]
            initial_matrix[j,k] = initial[i,j*(j+1)//2 + k +4]
            
            final_matrix[k,j] = final[i,j*(j+1)//2 + k + 4]
            final_matrix[j,k] = final[i,j*(j+1)//2 + k + 4]

    initial_eigenvs[i], _ = np.linalg.eig(initial_matrix)
    final_eigenvs[i], _ = np.linalg.eig(final_matrix)
    
#sort to obtain the same ordering in all rows

initial_eigenvs = np.sort(initial_eigenvs)

final_eigenvs = np.sort(final_eigenvs)

#estimate differences in percentage

deltas = (final_eigenvs - initial_eigenvs) / np.abs(initial_eigenvs)

#estimating width of distribution of percentual error

deltas_wo_tails = np.ma.masked_where((deltas < -0.1) | (deltas > 0.1),deltas)
widths = np.std(deltas_wo_tails, axis=0)

# Generate 5 plots in a 1x5 arrangement
f, axes = plt.subplots(1, 5)
f.set_size_inches(20, 15)
    
for i in range(5):
    axes[i].hist(deltas[:, i], range=(-0.1,0.1), bins=401, color="r")
    axes[i].annotate(f' width={widths[i]:.2}', xy=(0,0.97), xycoords='axes fraction')
    axes[i].set_title(f'Eigenvalue {i}')

plt.savefig('eigenv_deltas.png')
    
plt.show()

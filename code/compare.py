import numpy as np
import matplotlib.pyplot as plt

from neural_network import meansfilename

#import initial data

initial = np.loadtxt("data.txt")[::500]

#import final data

final = np.loadtxt("decompr_data.txt")

#define difference between initial data and after compression-decompression cycle

deltas = np.zeros(np.shape(initial))

deltas[:, 5:] = (final[:, 5:] - initial[:, 5:]) / np.abs(initial[:, 5:])

# Generate 16 plots in a 4x4 arrangement
f, axes = plt.subplots(4, 4)
f.set_size_inches(20, 15)

#estimating width of distribution of percentual error

deltas_wo_tails = np.ma.masked_where((deltas < -0.1) | (deltas > 0.1),deltas)
widths = np.std(deltas_wo_tails, axis=0)


for i in range(4):
    #print(f'i={i}')
    for j in range(4):
        k = i*4 + j + 4
        if k!=4:
            axes[i, j].hist(deltas[:, k], range=(-0.1,0.1), bins=401, color="r")
            axes[i, j].annotate(f' width={(widths[k]):.2}', xy=(0,0.9), xycoords='axes fraction')
            
plt.savefig('deltas.png')

plt.show()

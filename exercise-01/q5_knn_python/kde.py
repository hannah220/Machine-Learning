import numpy as np
import math
pi = np.pi

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    pos = np.arange(-5, 5.0, 0.1)
    estDensity = np.empty([0, 2], float)
    total = [0] * 100
    row = np.empty([0, 2], float)

    for i in range(0, 100, 1):
        for j in range(0, 100, 1):
            #sub.append(1/(math.sqrt(2 * pi) * h) * math.exp(-pow((pos[i] - samples[j]), 2)/(2 * pow(h, 2))))
            total[i] += 1/(math.sqrt(2 * pi) * h) * math.exp(-pow((pos[i] - samples[j]), 2)/(2 * pow(h, 2)))

        estDensity = np.vstack((estDensity, [pos[i], total[i] / 100]))
        
    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    return estDensity

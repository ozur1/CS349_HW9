import numpy as np
import scipy.stats

x = np.array([1, 1, 2, 2, 2, 3])
print(scipy.stats.mode(x)[0][0])
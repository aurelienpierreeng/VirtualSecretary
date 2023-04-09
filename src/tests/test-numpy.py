import numpy as np
import time
n = 2000
A = np.random.randn(n,n).astype('float64')
B = np.random.randn(n,n).astype('float64')
start_time = time.time()
nrm = np.linalg.norm(A*B)
print(" took {} seconds ".format(time.time() - start_time))
print(" norm = ",nrm)
print(np.__config__.show())

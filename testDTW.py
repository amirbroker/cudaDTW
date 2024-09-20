from DTW import dtwClc
import numpy as np
from numba import cuda
from datetime import datetime

seq = cuda.to_device(np.random.rand(50,80))
trg = cuda.to_device(np.random.rand(100,100))

startTime = datetime.now()
lenSeq = seq.shape[0]
lenTrg = trg.shape[0]

lenEachSeq = seq.shape[1]
lenEachTrg = trg.shape[1]

INDList = cuda.to_device(np.array([(i,j) for i in range(lenSeq) for j in range(lenTrg)]))

dtw_matrix = np.zeros((lenEachSeq+1, lenEachTrg+1))
dtw_matrix[0, :] = np.inf
dtw_matrix[:, 0] = np.inf
dtw_matrix[0, 0] = 0

dtw = cuda.to_device(np.zeros([lenSeq * lenTrg]))

lenDTW = dtw.shape[0]

dtwClc[lenDTW, 1024](seq, trg,dtw_matrix,lenEachSeq,lenEachTrg,lenDTW, INDList, dtw)
dtw = dtw.copy_to_host()

print(dtw)

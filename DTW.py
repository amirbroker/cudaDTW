from numba import cuda

@cuda.jit
def dtwClc(seq, trg,dtw_matrix,lenEachSeq,lenEachTrg,lenDTW, INDList, dtw):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x

    if ty > lenDTW:
        return

    id1, id2 = INDList[ty]

    for eachELeachTaq in range(1, lenEachTrg + 1):
        for eachELeachSeq in range(1, lenEachSeq + 1):
            if lenEachSeq == tx:
                cost = (seq[id1][0:lenEachSeq][eachELeachSeq - 1] - trg[id2][0:lenEachTrg][eachELeachTaq - 1])
                dtw_matrix[eachELeachSeq, eachELeachTaq] = cost + min(dtw_matrix[eachELeachSeq - 1, eachELeachTaq],  
                                                                      dtw_matrix[eachELeachSeq, eachELeachTaq - 1],    
                                                                      dtw_matrix[eachELeachSeq - 1, eachELeachTaq - 1]
                                                                      )

            cuda.syncthreads()
        dtw[ty] = dtw_matrix[-1, -1]

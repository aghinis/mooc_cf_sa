import numpy as np
import scipy.sparse as sp
from scipy import spatial
import os
import math

def dcg_at_k(r, k, method=0):
    """
    Desc:
        Discounted cumulative gain

    ------
    Input:
        r: list of real values (list)
        k: @k (int)
        method: 0 --> start with [1,1,...], 1 --> starts with [1,...]

    ------
    Output:
        dcg@k (float)
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def precision(recoms, test_set,k=10):

    num_users = len(recoms)
    test = test_set.A
    precision=[]
    for user in range(num_users):
        ranking = np.array(recoms[user])
        r=[]
        for item_id in ranking:
            real_value = test[user,item_id]
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        precision.append(np.mean(r))
    return np.mean(precision)

def ndcg(recoms, test_set,k=10,method=0):
    num_users = len(recoms)
    test = test_set.A
    ndcg=[]
    for user in range(num_users):
        ranking = np.array(recoms[user])
        r=[]
        for item_id in ranking:
            real_value = test[user,item_id]
            r.append(real_value)
        dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            ndcg.append(0.)
        else: ndcg.append(dcg_at_k(r, k, method) / dcg_max)
    return np.mean(ndcg)

def ndcg_time(recoms, test_set,test_time,k=10,method=0):
    num_users = len(recoms)
    test = test_set.toarray()
    test_time_c = test_time.copy().toarray()
    test_time_d = test_time.copy().toarray()
    for row, col in zip(*test_set.nonzero()):
        if test_set[row, col] == 1:
            test_time_c[row, col] = 0
        if test_set[row, col] == 2:
            test_time_d[row, col] = 0
    
    test_time_c = sp.csr_matrix(test_time_c)
    test_time_d = sp.csr_matrix(test_time_d)
    test_time_d = test_time_d/(test_time_d.max(axis=1).toarray())
    
    for i in range(test_time_c.shape[0]):
        # print(test_time_c[i,:].count_nonzero())
        # if(test_time_c[i,:].count_nonzero() == 0):
        #     print("OY") 
        min_time = np.min(test_time_c[i,:][test_time_c[i,:] != 0])
        for _,j in zip(*test_time_c[i].nonzero()):
            test_time_c[i,j]=(min_time/test_time_c[i,j])+1
    
    assert(test_time_d.multiply(test_time_c).nnz == 0)

    new_test = test_time_d + test_time_c
    new_test = new_test.toarray()
    ndcg=[]
    for user in range(num_users):
        ranking = np.array(recoms[user])
        r=[]
        for item_id in ranking:
            real_value = new_test[user,item_id]
            r.append(real_value)
        dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            ndcg.append(0.)
        else: ndcg.append(dcg_at_k(r, k, method) / dcg_max)
    return np.mean(ndcg)
    
def precision(recoms, test_set,k=10):

    num_users = len(recoms)
    test = test_set.A
    precision=[]
    for user in range(num_users):
        ranking = np.array(recoms[user])
        r=[]
        for item_id in ranking:
            real_value = test[user,item_id]
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        precision.append(np.mean(r))
    return np.mean(precision)

def recall(recoms,test_set,k=10):

    num_users = len(recoms)
    recall=[]
    test = test_set.A
    for user in range(num_users):
        ranking = np.array(recoms[user])
        r=[]
        relevant = test_set[user,:].count_nonzero()
        if relevant == 0 or math.isnan(relevant):
            continue
        for item_id in ranking:
            real_value = test[user,item_id]
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        recall.append(np.sum(r)/relevant)
    return np.mean(recall)

def MAP(recoms,test_set,k=10):

    num_users = len(recoms)
    test = test_set.A
    ap=[]
    for user in range(num_users):
        ranking = np.array(recoms[user])
        r=[]
        dg =[]
        relevant = test_set[user,:].count_nonzero()
        if relevant == 0 or math.isnan(relevant):
            continue
        for item_id in ranking:
            i = 1
            real_value = test[user,item_id]
            dg.append(1/i)
            i+=1
            if real_value==0:
                r.append(real_value)
            else:
                r.append(1)
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        dg = np.asarray(dg)[:k] != 0
        if dg.size != k:
            raise ValueError('Relevance score length < k')
        ap.append(np.dot(r,dg)/relevant)
    return np.mean(ap)

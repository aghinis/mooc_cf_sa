import pandas as pd
import scipy.sparse as sps
import numpy as np



def TF_IDF(dataMatrix):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :return:
    """

    assert np.all(np.isfinite(dataMatrix.data)), \
        "TF_IDF: Data matrix contains {} non finite values.".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

    assert np.all(dataMatrix.data >= 0.0),\
        "TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.".format(np.sum(dataMatrix.data < 0.0))

    # TFIDF each row of a sparse amtrix
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])

    # calculate IDF
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # apply TF-IDF adjustment
    dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

    return dataMatrix.tocsr()

def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """


    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)

    elif format == 'npy':
        if sps.issparse(X):
            return X.toarray().astype(dtype)
        else:
            return np.array(X)

    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)

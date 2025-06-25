import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import GridSearchCV
# from skopt import BayesSearchCV
from sklearn.preprocessing import MinMaxScaler

def librec_att_setter(model,p_dict):
    for i,v in p_dict.items():
        model.__setattr__(i,v)

def prep_data_librec(df):
    df.completed = 1
    df.rename(columns={'username':'user','course_id':'item','completed':'label'},inplace=True)
    return df

def transform_dict(original_dict,train, key_map, value_map):
    transformed_dict = {}
    for key, value_list in original_dict.items():
        # Map the key using key_map, or keep the original if not found
        new_key = key_map.get(key)
        training_inter = train[new_key].nonzero()[1]
        # Map each value in the list using value_map, or keep the original if not found
        new_values = [value_map.get(val) for val in value_list]
        new_values_no_train = [x for x in new_values if x not in training_inter]
        transformed_dict[new_key] = new_values_no_train
        sorted_dict = dict(sorted(transformed_dict.items()))
    return sorted_dict


def recom_librec(model, train_df,train,rid_idx, cid_idx):
    users = train_df.user.unique()
    n_items = len(train_df.item.unique())
    recoms_dict = model.recommend_user(user = users,n_rec=n_items)
    recomms = list(transform_dict(recoms_dict,train,rid_idx,cid_idx).values())
    return recomms

def re_ranker(first_lists,second_lists,first_k,second_k):

    initial_lists = [i[:first_k] for i in first_lists]
    final_lists = [[i for i in second_lists[j] if i in initial_lists[j]][:second_k] for j in range(len(second_lists))]

    return final_lists
    
def recom_lf (model, train_set):
    recom = []
    num_users,num_items=train_set.shape
    items = np.array(range(train_set.shape[1]))
    for user in range(num_users):
        training_inter = train_set[user].nonzero()[1]
        scores = model.predict(user,items,num_threads=1)
        ranking = np.argsort(-scores)
        ranking = [x for x in ranking if x not in training_inter]
        recom.append(ranking)
    return recom

def recom_knn (model, train_set):
    num_users,_=train_set.shape
    recoms = model.recommend(range(num_users))
    return recoms

def threshold_interactions_df_mooc(df, row_name, col_name, min_enrollments, min_completions):
    # user filtering
    tmp_df = df.groupby([row_name,'completed'])[col_name].count().reset_index().pivot(index=row_name,values=col_name,columns='completed').fillna(0)
    tmp_df['total'] = tmp_df[1]+ tmp_df[2]
    idx = tmp_df[(tmp_df['total']>=min_enrollments) & (tmp_df[2]>=min_completions)].index.tolist()
    # course filtering
    tmp_df2 = df.groupby([col_name,'completed'])[row_name].count().reset_index().pivot(index=col_name,values=row_name,columns='completed').fillna(0)
    tmp_df2['total'] = tmp_df2[1]+ tmp_df2[2]
    idx2 = tmp_df2[(tmp_df2['total']>=min_enrollments) & (tmp_df2[2]>=min_completions)].index.tolist()


    # filter on both
    final_df = df[(df[row_name].isin(idx)) & (df[col_name].isin(idx2))]
    n_rows = final_df[row_name].unique().shape[0]
    n_cols = final_df[col_name].unique().shape[0]
    sparsity = float(final_df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    return final_df


def train_test_sp(interactions, split_count, min_completed, fraction=None):

    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        # test_interactions = np.random.choice(np.where(interactions.getrow(user).todense()==2)[1],
        #                                 size=split_count,
        #                                 replace=False)
        completed_courses = np.where(interactions.getrow(user).todense() == 2)[1]
        dropout_courses = np.where(interactions.getrow(user).todense() == 1)[1]
        all_interactions = np.concatenate((completed_courses,dropout_courses))
        test_interactions_completed = np.random.choice(completed_courses,
                                                    size=min_completed,
                                                    replace=False)
        drop_idx = np.where(np.isin(all_interactions,test_interactions_completed))
        remaining_interactions = np.random.choice(np.delete(all_interactions,drop_idx),
                                                split_count-min_completed,
                                                replace=False)
        test_interactions = np.concatenate((test_interactions_completed,remaining_interactions))
        test_interactions

        train[user, test_interactions] = 0.
        # These are just 1.0 right now
        test[user, test_interactions] = interactions[user, test_interactions]

    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index

def train_test(interactions, split_count, fraction=None):
    """
    Desc:
        Using this function, we split avaialble data to train and test set.
    ------
    Input:
        interactions : interaction between users and streams (scipy.sparse matrix)
        split_count : number of interactions per user to move from training to test set (int)
        fraction : fraction of users to split their interactions train/test. If None, then all users (float)
    ------
    Output:
        train_set (scipy.sparse matrix)
        test_set (scipy.sparse matrix)
        user_index
    """
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        if train.getrow(user).count_nonzero()<2:
            continue
        test_interactions = np.random.choice(interactions.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        train[user, test_interactions] = 0.
        test[user, test_interactions] = interactions[user, test_interactions]

    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index


def get_df_matrix_mappings(df, row_name, col_name):
    """
    Desc:
        Using this function, we are able to get mappings between original indexes and new (reset) indexes

    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user_id (str)
        col_nam: column name referring to stream_slug (str)

    ------
    Output:
        rid_to_idx: a dictionary contains mapping between real row ids and new indexes (dict)
        idx_to_rid: a dictionary contains mapping between new indexes and real row ids (dict)
        cid_to_idx: a dictionary contains mapping between real column ids and new indexes (dict)
        idx_to_cid: a dictionary contains mapping between new indexes and real column ids (dict)
    """

    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_mat(df, row_name, col_name,value):
    """
    Desc:
        Using this function, we transfrom the interaction matrix to scipy.sparse matrix

    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user_id (str)
        col_nam: column name referring to stream_slug (str)
        value: the value of interaction between row and column

    ------
    Output:
        interactions: Sparse matrix contains user and streams interactions (sparse csr)
        rid_to_idx: a dictionary contains mapping between real row ids and new indexes (dict)
        idx_to_rid: a dictionary contains mapping between new indexes and real row ids (dict)
        cid_to_idx: a dictionary contains mapping between real column ids and new indexes (dict)
        idx_to_cid: a dictionary contains mapping between new indexes and real column ids (dict)
    """


    rid_to_idx, idx_to_rid,cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,row_name,col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = df[value]
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def matrix_to_df(x,r,c):
    d = []
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        d.append({'username':r[i],'course_id':c[j],'completed':v})
    return pd.DataFrame.from_dict(d)

def matrix_to_df_2(x,r,c):
    d = []
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        d.append({'user_id':r[i],'item_id':c[j],'rating':v})

    for i,j in enumerate(np.asarray(cx.sum(0))[0]):
        if j ==0:
            d.append({'user_id':r[0],'item_id':c[i],'rating':0})
    return pd.DataFrame.from_dict(d)

def set_intersection(a,b):
    return list(set(a).intersection(set(b)))

def get_0_and_p_index(data):
    num_users,num_items=data.shape
    user_nonzero = []
    user_zero = []
    for i in range(data[:,0].shape[0]):
       p_idxes = data[i,:].nonzero()[1]
       j_idx = np.where(data.A[i]==0)[0]
       user_nonzero.append(p_idxes)
       user_zero.append(j_idx)
    return user_nonzero,user_zero

def set_diff(a,b):
    return list(set(a)-set(b))

def matrix_completion(train_set,trained_model,model_name):
    if model_name == "SLIM" or model_name == "EASE":
        matrix = np.matmul(train_set.A,trained_model.W_sparse.A)
    elif model_name == "SVD":
        matrix = np.matmul(trained_model.USER_factors,trained_model.ITEM_factors.T)
    elif model_name == "MVAE":
        matrix = trained_model._compute_item_score(range(train_set.shape[0]))
        scaler = MinMaxScaler()
        matrix = scaler.fit_transform(matrix)
        matrix = np.where(matrix==0,0.0001,matrix)
    cx = train_set.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        matrix[i,j]=v
    return matrix

def Grid_tune(X,Y,model,params,c_v=5,jobs=10,score='neg_mean_squared_error'):
    reg_bay = GridSearchCV(estimator=model,
                    param_grid=params,
                    cv=c_v,
                    n_jobs=jobs,
                    scoring=score,
                    verbose=True)
    reg_bay.fit(X, Y)
    return reg_bay, reg_bay.best_params_

def Bayes_tune(X,Y,model,params,niter=10,c_v=5,jobs=10,score='neg_mean_squared_error'):

    reg_bay = BayesSearchCV(estimator=model,
                    search_spaces=params,
                    n_iter=niter,
                    cv=c_v,
                    n_jobs=jobs,
                    scoring=score,
                    verbose=True)
    reg_bay.fit(X, Y)
    return reg_bay, reg_bay.best_params_

def compute_user_profile (matrix, item_features):
    '''
    content: input from article emdeddings containing labelencoders, dataframe and embeddings
    indexes: article ids in the session
    '''
    Mcsr = matrix.tocsr()
    non_zeros = np.split(Mcsr.indices, Mcsr.indptr)
    non_zero_list = non_zeros[1:-1]
    profiles = []
    for user in range(len(non_zero_list)):
        embeddings = item_features[non_zero_list[user]]
        aggregate = np.mean(embeddings, axis=0)
        profiles.append(aggregate)
    return np.asarray(profiles)

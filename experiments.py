import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import os
import ast
from functools import partial
from pandasql import sqldf
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_density, stat_ecdf, scale_color_discrete, theme

from sklearn.decomposition import PCA
import xgboost as xgb
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import sksurv
from sksurv.metrics import concordance_index_censored
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from baselines.knn import ItemKNNCFRecommender,UserKNNCFRecommender

from helpers.measures import MAP, recall, ndcg, precision,ndcg_time
from helpers.utils import train_test_sp, train_test, df_to_mat ,threshold_interactions_df_mooc, matrix_to_df, recom_knn, re_ranker
import helpers.tunning_param as tp
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import copy
from tqdm.auto import tqdm
from sksurv.metrics import concordance_index_censored
import logging
import os
import pickle 
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from baselines.IALSRecommender import IALSRecommender
from baselines.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from baselines.EASE_R_Recommender import EASE_R_Recommender
from baselines.PureSVDRecommender import PureSVDRecommender
from baselines.NMFRecommender import NMFRecommender

def log_print(*args):
    message = " ".join(map(str, args))
    print(message)  # Print to console
    logging.info(message)  # Log to file

def min_max_normalize(df, groupby_cols, normalize_col):
    """
    Perform min-max normalization on a DataFrame column after grouping by specified columns.

    Parameters:
    df (DataFrame): Input DataFrame.
    groupby_cols (list): List of column names to group by.
    normalize_col (str): Name of the column to normalize.

    Returns:
    DataFrame: DataFrame with normalized column.
    """
    # Group by specified columns
    df_copy = copy.deepcopy(df)
    grouped_df = df_copy.groupby(groupby_cols)

    # Calculate min-max normalization for each group
    min_values = grouped_df[normalize_col].transform('min')
    max_values = grouped_df[normalize_col].transform('max')

    # Avoid division by zero
    min_max_normalized = (df_copy[normalize_col] - min_values) / (max_values - min_values).replace(0, 1)

    # Replace the column in the original DataFrame with normalized values
    df_copy[normalize_col] = min_max_normalized

    return df_copy

def rank_sum_sort(list_1, list_2):
    # Create a dictionary to store the sum of ranks for each element
    rank_sum = {}

    for element in list_1:
        # Find the index (rank) of the element in both lists and sum them
        rank_sum[element] = list_1.index(element) + list_2.index(element)

    # Sort the elements based on the sum of ranks
    sorted_elements = sorted(rank_sum, key=rank_sum.get)

    return sorted_elements

def replace_with_max(df, group_col, binary_col, replace_col):
    """
    Replace values in a DataFrame column where the value of a binary column is 2
    with the maximum value from records where the binary column is 1,
    within each group formed by the group column.

    Parameters:
    df (DataFrame): Input DataFrame.
    group_col (str): Name of the column to group by.
    binary_col (str): Name of the binary column containing values 1 and 2.
    replace_col (str): Name of the column to replace values in.

    Returns:
    DataFrame: DataFrame with values replaced.
    """
    # Group by the specified column
    grouped_df = df.groupby(group_col)

    # Iterate over groups
    for name, group in grouped_df:
        # Find max value where binary column is 1
        max_val = group.loc[group[binary_col] == 1, replace_col].max()
        # Replace values in replace_col where binary_col is 2
        df.loc[(df[group_col] == name) & (df[binary_col] == 2), replace_col] = max_val

    return df

def load_dataset(name):
    if name == 'X':
        data=pd.read_csv("xuentangx_processed_v2.csv")

        data["completed"] = data["completed"] + 1
        data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.total_seconds()/86400
    elif name == 'KDD':
        data=pd.read_csv("kddcup_processed.csv")

        data["completed"] = data["completed"] + 1
        data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.total_seconds()/86400
    elif name == 'Canvas':
        cols_to_use = ['username','course_id','days_spent','completed']
        data=pd.read_csv("canvas_preprocessed.csv",index_col=False,usecols=cols_to_use)[cols_to_use]
    return data

def df_to_matrix(df):

    bin_mtx  = pd.crosstab(df['username'],df['course_id'], df['completed'], aggfunc='mean', dropna=False).fillna(0)
    time_mtx = pd.crosstab(df['username'],df['course_id'], df['days_spent'], aggfunc='mean', dropna=False).fillna(0)

    return bin_mtx, time_mtx

# generate one-hot encoding intances
def one_hot(users, courses):

    u_one_hot = pd.get_dummies(users)
    c_one_hot = pd.get_dummies(courses)

    return u_one_hot, c_one_hot

# get interaction features
def get_interaction_feats(b_mtx, t_mtx):

    u_b_int = b_mtx.copy(deep=True)
    u_t_int = t_mtx.copy(deep=True)

    c_b_int = b_mtx.copy(deep=True).T
    c_t_int = t_mtx.copy(deep=True).T

    return u_b_int, u_t_int, c_b_int, c_t_int

# get xgb survival outcome
def surv_xgb(y):

    y_xgb = [t[0] if t[1]==1 else -t[0] for t in y.values]

    return y_xgb

# train model
def train_xgb(df_x, df_y):

    y_s = surv_xgb(df_y)
    x_train_xgb     =   xgb.DMatrix(df_x, label=y_s)

    param       = {'objective': 'survival:cox'}
    xgb_model   = xgb.train(param, x_train_xgb)

    return xgb_model

def tune_and_fit_coxnet(X, y):
    """
    Perform parameter tuning for Coxnet model and fit the final model.

    Parameters:
    X (DataFrame): The covariates.
    y (structured array): The structured array with fields 'Status' and 'Survival_in_days'.

    Returns:
    CoxnetSurvivalAnalysis: Fitted Coxnet model with the best parameters.
    """

    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100))
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(X, y)

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=1,
    ).fit(X, y)


    coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
    coxnet_pred.set_params(**gcv.best_params_)
    coxnet_pred.fit(X, y)

    return coxnet_pred

# get unlabelled data
def get_unlabelled_interactions(bin_mtx, u_b_int, u_t_int, c_b_int, c_t_int, n):

    unl_df = []
    unl_df_names = []
    l_rows, l_cols = bin_mtx.shape

    iters = 0
    while len(unl_df) < n and iters < n*100:
        row = random.randint(0, l_rows-1)
        col = random.randint(0, l_cols-1)

        if bin_mtx.iloc[row, col] == 0:
            stud_id = bin_mtx.iloc[row, :].name
            cours_id = bin_mtx.columns[col]

            new_row_u = np.hstack((u_b_int.loc[stud_id, :].values, u_t_int.loc[stud_id, :].values))
            new_row_c = np.hstack((c_b_int.loc[cours_id, :].values, c_t_int.loc[cours_id, :].values))

            new_row   = np.hstack((new_row_u, new_row_c))
            names     = np.hstack((stud_id, cours_id))

            unl_df.append(new_row)
            unl_df_names.append(names)

        iters +=1

    return pd.DataFrame(unl_df), pd.DataFrame(unl_df_names)

def get_all_unlabelled_interactions(bin_mtx, u_b_int, u_t_int, c_b_int, c_t_int):

    unl_df = []
    unl_df_names = []
    l_rows, l_cols = bin_mtx.shape

    for i in range(l_rows):
        for j in range(l_cols):
            if bin_mtx.iloc[i,j] == 0:
                stud_id = bin_mtx.iloc[i, :].name
                cours_id = bin_mtx.columns[j]

                new_row_u = np.hstack((u_b_int.loc[stud_id, :].values, u_t_int.loc[stud_id, :].values))
                new_row_c = np.hstack((c_b_int.loc[cours_id, :].values, c_t_int.loc[cours_id, :].values))

                new_row   = np.hstack((new_row_u, new_row_c))
                names     = np.hstack((stud_id, cours_id))

                # unl_df = np.append(unl_df,new_row)
                # unl_df_names = np.append(unl_df_names,names)

                unl_df.append(new_row)
                unl_df_names.append(names)

    return pd.DataFrame(unl_df), pd.DataFrame(unl_df_names)

def get_all_unlabelled_interactions_opt(bin_mtx, u_b_int, u_t_int, c_b_int, c_t_int):
    # Replace non-zero values with NaN and stack
    stacked_bin_mtx = bin_mtx.where(bin_mtx == 0).stack(level=0)

    # Preallocate arrays for efficiency
    num_rows = len(stacked_bin_mtx)
    num_cols = u_b_int.shape[1] + u_t_int.shape[1] + c_b_int.shape[1] + c_t_int.shape[1]
    unl_df_array = np.empty((num_rows, num_cols), dtype=float)
    unl_df_names = np.empty((num_rows, 2), dtype=object)

    # Process data without explicit Python loops
    for idx, (stud_id, cours_id) in enumerate(stacked_bin_mtx.index):
        u_b_vals = u_b_int.loc[stud_id].values
        u_t_vals = u_t_int.loc[stud_id].values
        c_b_vals = c_b_int.loc[cours_id].values
        c_t_vals = c_t_int.loc[cours_id].values

        unl_df_array[idx, :] = np.hstack((u_b_vals, u_t_vals, c_b_vals, c_t_vals))
        unl_df_names[idx, :] = [stud_id, cours_id]

    # Convert arrays to DataFrame in one go
    return pd.DataFrame(unl_df_array), pd.DataFrame(unl_df_names, columns=['username', 'course_id'])

def get_reliable_predictions(new_preds, n_int):

    final_preds = []

    for user in new_preds.username.unique():

        tmp = new_preds[new_preds.username == user].sort_values(['pred'], ascending=False)
        tmp = tmp.iloc[0:n_int, :]

        final_preds.append(tmp)

    df_final_preds = pd.DataFrame()

    for df in final_preds:
        df_final_preds = df_final_preds.append(df, ignore_index=True)

    return df_final_preds

def tune_rsf(dataset,X,y,event):
    fname = f'trained_models/{dataset}_{event}_rsf.pkl'
    if os.path.exists(fname):
        with open(fname, 'rb') as file:
            model = pickle.load(file)
    else:
        tree_params = {
        'n_estimators': scope.int(hp.quniform('n_estimators',25,100,q=1)),
        'max_features': 'sqrt',
        'min_samples_leaf' : scope.int(hp.quniform('min_samples_leaf',10,20,q=1)),
        'min_samples_split' : scope.int(hp.quniform('min_samples_split',10,20,q=1)),
        'max_depth': scope.int(hp.quniform('max_depth',2,12,q=1)),
        'max_samples' : 0.5,
        'n_jobs' : 10
        }
        def f(params):
            rsf_tree = RandomSurvivalForest(**params)
            score = cross_val_score(rsf_tree,X,y, cv=5,n_jobs=-1).mean() *-1
            return score
        trials = Trials()
        result = fmin(
            fn=f,                          
            space=tree_params,   
            algo=tpe.suggest,              
            max_evals=100,                  
            trials=trials                   
        )
        result['n_estimators'] = int(result['n_estimators'])
        result['min_samples_leaf'] = int(result['min_samples_leaf'])
        result['min_samples_split'] = int(result['min_samples_split'])
        result['max_depth'] = int(result['max_depth'])
        result['max_features'] = 'sqrt'
        result['n_jobs'] = 10
        result['max_samples'] = 0.5
        model = RandomSurvivalForest(**result).fit(X,y)
        with open(fname,'wb') as file:
            pickle.dump(model,file)

    return model

def tune_boosted(dataset,X,y,event):
    fname = f'trained_models/{dataset}_{event}_xgb.pkl'
    if os.path.exists(fname):
        with open(fname, 'rb') as file:
            model = pickle.load(file)
    else:
        tree_params = {
        'learning_rate': hp.uniform('learning_rate',0.1,1),
        'n_estimators': scope.int(hp.quniform('n_estimators',25,200,q=1)),
        'max_features': 'sqrt',
        'min_samples_leaf' : scope.int(hp.quniform('min_samples_leaf',5,20,q=1)),
        'min_samples_split' : scope.int(hp.quniform('min_samples_split',5,20,q=1)),
        'max_depth': scope.int(hp.quniform('max_depth',2,20,q=1)),
        'subsample' : 0.5 
        }
        def f(params):
            rsf_tree = GradientBoostingSurvivalAnalysis(**params)
            score = cross_val_score(rsf_tree,X,y, cv=5,n_jobs=-1).mean() *-1
            return score
        trials = Trials()
        result = fmin(
            fn=f,                          
            space=tree_params,   
            algo=tpe.suggest,              
            max_evals=100,                  
            trials=trials                   
        )
        result['n_estimators'] = int(result['n_estimators'])
        result['min_samples_leaf'] = int(result['min_samples_leaf'])
        result['min_samples_split'] = int(result['min_samples_split'])
        result['max_depth'] = int(result['max_depth'])
        result['max_features'] = 'sqrt'
        result['n_jobs'] = 10
        result['subsample'] = 0.5
        model = GradientBoostingSurvivalAnalysis(**result).fit(X,y)
        with open(fname,'wb') as file:
            pickle.dump(model,file)
    return model


def run_all_pca(dataset,split_count=3,min_completed=1, normalize_time=True):
    log_print(f"\n Starting: {dataset} split_count={split_count}, min_completed = {min_completed}, normalized_time={normalize_time}, PCA on intercations")
    num_folds_tunning = 3
    tunning = False
    njobs = -1
    random_seed = 1
    #############################
    random.seed(random_seed)
    np.random.seed(random_seed)
    data = load_dataset(dataset)
    if dataset == 'X':
        data['course_id'] = data['course_id'].replace(course_mapping_dict)
        data = sqldf('''
        select username, course_id, max(days_spent) as days_spent, max(completed) as completed from data group by 1,2 
        ''')
    elif dataset == 'Canvas':
        data = sqldf('''
        select username, course_id, max(days_spent) as days_spent, max(completed) as completed from data group by 1,2 
        ''')

    print(len(data['course_id'].unique()))
    if dataset =='Canvas':
        data = threshold_interactions_df_mooc(data, "username","course_id", 5, 3)
    else:
        data = threshold_interactions_df_mooc(data, "username","course_id", 5, 3)

    interactions,rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid = df_to_mat(data,"username","course_id","completed")
    train_set,test_set,users_set = train_test_sp(interactions,split_count=split_count,min_completed=min_completed)

    # get time matrix for ndcg_time
    test_time = test_set.copy()
    for row, col in zip(*test_set.nonzero()):
        test_time[row,col] = data[(data['username']==idx_to_rid[row]) & (data['course_id']==idx_to_cid[col])]['days_spent'].array[0]
        if test_time[row,col] == 0:
            test_time[row,col] = 0.001
            
    print("number of unlabeld for user 0:", train_set.shape[1] - len(train_set[0].nonzero()[0]))
    print("number of training inter: ",len(train_set.nonzero()[0]))
    print("number of testing inter: ",train_set.shape[0]*train_set.shape[1] - len(train_set.nonzero()[0]))

    if dataset =='X': 
        feature_course = pd.read_csv('x_features_formatted.csv', index_col=0).drop(columns=['namevector', 'aboutvector'])
        feature_course['id'] = feature_course['id'].replace(course_mapping_dict)
        feature_course = feature_course.rename(columns={'id': 'course_id'}).drop(columns='coursetype_0')
        feature_course =feature_course.groupby('course_id').mean()

    train_df = matrix_to_df(train_set,idx_to_rid, idx_to_cid)
    train_df_time = pd.merge(train_df, data, on = ['username','course_id','completed'], how='left')

    #
    df = train_df_time.dropna()
    bin_mtx, time_mtx = df_to_matrix(df)
    u_b_int, u_t_int, c_b_int, c_t_int = get_interaction_feats(bin_mtx, time_mtx)
    u_b_int.columns = [f'{col}_u_b' for col in u_b_int.columns]
    u_t_int.columns = [f'{col}_u_t' for col in u_t_int.columns]
    c_b_int.columns = [f'{col}_c_b' for col in c_b_int.columns]
    c_t_int.columns = [f'{col}_c_t' for col in c_t_int.columns]

    df2 = min_max_normalize(df,['course_id','completed'],'days_spent')
    bin_mtx, time_mtx = df_to_matrix(df2)
    u_b_int, u_t_int, c_b_int, c_t_int = get_interaction_feats(bin_mtx, time_mtx)
    u_b_int.columns = [f'{col}_u_b' for col in u_b_int.columns]
    u_t_int.columns = [f'{col}_u_t' for col in u_t_int.columns]
    c_b_int.columns = [f'{col}_c_b' for col in c_b_int.columns]
    c_t_int.columns = [f'{col}_c_t' for col in c_t_int.columns]

    user_info = df2.loc[:, ['username', 'course_id', 'days_spent', 'completed']]
    user_features = sqldf('''
        select username
        , courses_taken
        , pct_completed
        , avg_days_to_completion
        , avg_days_to_dropout
        , coalesce(relative_time_overall,0) as relative_time_overall
        from (select username
        , count(*) as courses_taken
        , avg(case when completed = 2 then 1 else 0 end) as pct_completed
        , coalesce(avg(case when completed = 2 then days_spent end),0) as avg_days_to_completion 
        , coalesce(avg(case when completed = 1 then days_spent end),0) as avg_days_to_dropout
        from user_info
        group by 1)
        join (
        select username,
        avg(relative_time) as relative_time_overall
        from (select username
        , coalesce(days_spent/avg(days_spent) over(partition by course_id),0) as relative_time
        from user_info)
        group by 1
        order by 1
        ) using(username)
        order by username 
        ''').set_index('username')
    if normalize_time:
        df_with_feats1 = df2.loc[:, ['username', 'course_id', 'days_spent', 'completed']].join(u_b_int, on='username', how='left')
    else:
        df_with_feats1 = df.loc[:, ['username', 'course_id', 'days_spent', 'completed']].join(u_b_int, on='username', how='left')
    df_with_feats1 = df_with_feats1.join(u_t_int, on='username', how='left')
    df_with_feats1 = df_with_feats1.join(c_b_int, on='course_id', how='left')
    df_with_feats1 = df_with_feats1.join(c_t_int, on='course_id', how='left')
    if dataset=='X':
        df_with_feats1 = df_with_feats1.merge(feature_course, on='course_id')
    df_with_feats1 = df_with_feats1.merge(user_features, on='username')

    X_train = df_with_feats1.drop(columns=['username', 'course_id', 'days_spent', 'completed'])
    y_train = df_with_feats1.loc[:, ['days_spent', 'completed']]
    X_train.head()

    pca_pipeline_1 = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=0.8, svd_solver = 'full'))
    ])        
    preprocessor = ColumnTransformer(
        transformers=[
            ('pca1', pca_pipeline_1, u_b_int.columns.tolist()+u_t_int.columns.tolist()+ c_b_int.columns.tolist()+c_t_int.columns.tolist()),
            ('user_scaler', StandardScaler(), user_features.columns.tolist()),

        ]
    )
    transformed_data = preprocessor.fit_transform(X_train)
    pca1_names = [f'PCA_bin{i+1}' for i in range(len(preprocessor.named_transformers_['pca1'].named_steps['pca'].explained_variance_))]
    final_column_names = pca1_names  + user_features.columns.tolist() 

    transformed_df = pd.DataFrame(transformed_data, columns=final_column_names)
    # define dropout
    y_test = [(True,t[0]) if t[1]==1 else (False,t[0]) for t in y_train.values]
    y_test = np.array(y_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    # define completion
    y_test_2 = [(True,t[0]) if t[1]==2 else (False,t[0]) for t in y_train.values]
    y_test_2 = np.array(y_test_2, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    ranking_results = list()
    surv_models = ['Coxnet','rsf','XGb']
    for surv_model in surv_models: 
        log_print(f'Starting {surv_model}')
        if surv_model == 'Coxnet': 
            print("training dropout")
            model_dropout = tune_and_fit_coxnet(transformed_df, y_test)
            print("Training Completion")
            model_completion = tune_and_fit_coxnet(transformed_df, y_test_2)
        elif surv_model =='rsf':
            print("training dropout")
            model_dropout = tune_rsf(dataset,transformed_df, y_test,'dropout')
            print("Training Completion")
            model_completion = tune_rsf(dataset,transformed_df, y_test_2, 'completion')
        elif surv_model == 'XGb':
            print("training dropout")
            model_dropout = tune_boosted(dataset,transformed_df, y_test,'dropout')
            print("Training Completion")
            model_completion = tune_boosted(dataset,transformed_df, y_test,'completion')
        log_print(f'Dropout C-index {surv_model}: {cross_val_score(model_dropout,transformed_df, y_test,cv=5, n_jobs=-1).mean()}')
        log_print(f'Completion C-index {surv_model}: {cross_val_score(model_completion,transformed_df, y_test,cv=5, n_jobs=-1).mean()}')
        print(model_dropout)
        stacked_bin_mtx = bin_mtx.where(bin_mtx == 0).stack(level=0)
        batch_size = 10000
        num_rows = len(stacked_bin_mtx)
        num_batches = (num_rows + batch_size - 1) // batch_size 

        def process_batch(batch):
            num_rows = len(batch)
            num_cols = X_train.shape[1] 
            batch_array = np.empty((num_rows, num_cols), dtype=float)
            
            for idx, (stud_id, cours_id) in enumerate(batch):
                u_b_vals = u_b_int.loc[stud_id].values
                u_t_vals = u_t_int.loc[stud_id].values
                c_b_vals = c_b_int.loc[cours_id].values
                c_t_vals = c_t_int.loc[cours_id].values
                if dataset == 'X':
                    c_feats = feature_course.loc[cours_id].values
                user_feat = user_features.loc[stud_id].values
                
                if dataset =='X': 
                    batch_array[idx, :] = np.hstack((u_b_vals, u_t_vals, c_b_vals, c_t_vals, c_feats,user_feat))
                else:
                    batch_array[idx, :] = np.hstack((u_b_vals, u_t_vals, c_b_vals, c_t_vals,user_feat))
            
            batch_df = pd.DataFrame(batch_array, columns=X_train.columns)
            transformed_batch_df = pd.DataFrame(preprocessor.transform(batch_df))
            return transformed_batch_df

        batches = [stacked_bin_mtx.index[i:i + batch_size] for i in range(0, num_rows, batch_size)]
        res = [process_batch(batch) for batch in tqdm(batches, total=num_batches,position=0, leave=True)]
        final_df = pd.concat(res, ignore_index=True)
        batch_indices_list = [range(i, min(i + batch_size, num_rows)) for i in range(0, num_rows, batch_size)]
        pred_unl_dropout = np.concatenate([model_dropout.predict(final_df.iloc[i,:]) for i in batch_indices_list])
        pred_unl_completion = np.concatenate([model_completion.predict(final_df.iloc[i,:]) for i in batch_indices_list])
        
        new_unl_df_names = stacked_bin_mtx.reset_index().drop(columns=0)
        new_unl_df_names.loc[:, 'predicted_dropout'] = pred_unl_dropout
        new_unl_df_names.loc[:, 'predicted_completion'] = pred_unl_completion
        new_unl_df_names.loc[:, 'completed'] = np.ones((len(new_unl_df_names, )))

        print("unlabeled for 0 ",len(new_unl_df_names.loc[new_unl_df_names['username']==idx_to_rid[0]]))
        if dataset =='KDD':
            new_unl_df_names['username'] = new_unl_df_names['username']
        else:
            new_unl_df_names['username'] = new_unl_df_names['username'].astype('int')
        new_unl_df_names['course_id'] = new_unl_df_names['course_id'].replace(cid_to_idx)
        new_unl_df_names['username'] = new_unl_df_names['username'].replace(rid_to_idx)

        recomm_surv_1 = new_unl_df_names.sort_values(by=['predicted_dropout'], ascending=[True]).groupby('username')['course_id'].apply(list)
        recomm_surv_1 = recomm_surv_1.to_list()
        recomm_surv_2 = new_unl_df_names.sort_values(by=['predicted_completion'], ascending=[False]).groupby('username')['course_id'].apply(list)
        recomm_surv_2 = recomm_surv_2.to_list()

        recomm_surv_3=[]
        for i in range(len(recomm_surv_2)):
            recomm_surv_3.append(rank_sum_sort(recomm_surv_1[i],recomm_surv_2[i]))

        print("ndcg with Dropout: ",ndcg(recomm_surv_1,test_set,k=3))
        print("ndcg with Completion: ",ndcg(recomm_surv_2,test_set,k=3))
        print("ndcg with both: ",ndcg(recomm_surv_3,test_set,k=3))

        print("unlabeled for 0 ",len(new_unl_df_names.loc[new_unl_df_names['username']==0]))

        scores = {
                "EASE":{"ndcg":[],"h_param":['topK', 'l2_norm'],"h_param_range":[[None],(1e0, 1e7)],'best_param':{"X":[None, 95109.6942962282],"Canvas":[None, 9325573.660829231],"KDD":[None, 2540.6584595004156]}},
                "UKNN":{"ndcg":[],"h_param":['topK', 'shrink'],"h_param_range":[(20, 800),(0,1000)],'best_param':{"X":[301, 178],"Canvas":[128, 8],"KDD":[488, 907]}},
                "IKNN":{"ndcg":[],"h_param":['topK', 'shrink'],"h_param_range":[(20, 800),(0,1000)],'best_param':{"X":[70, 350],"Canvas":[789, 793],"KDD":[37, 194]}},
                "SVD":{"ndcg":[],"h_param":['num_factors','random_seed'],"h_param_range":[(3, 300),[int(random_seed)]],'best_param':{"X":[5, 1],"Canvas":[8, 1],"KDD":[3, 1]}},
                "NMF":{"ndcg":[],"h_param":['num_factors','l1_ratio'],"h_param_range":[(10, 300),(0.1,0.9)],'best_param':{"X":[202, 0.21397251366409453],"Canvas":[242, 0.2322551413506923],"KDD":[43, 0.8460852461396278]}},
                "SLIM":{"ndcg":[],"h_param":['topK', 'l1_ratio','alpha'],"h_param_range":[(5, 600),(1e-5,1.0),(1e-3, 1.0)],'best_param': {"X":[380, 0.00023022491889188525, 0.31065433709916135],"Canvas":[486, 0.5934988381321606, 0.00590093297649861],"KDD":[321, 0.03911733924386367, 0.17916129385389876]}},
                "IALS":{"ndcg":[],"h_param":['epochs','num_factors','reg'],"h_param_range":[(10, 200),(10,100),(1e-5, 1e-1)],'best_param':{"X":[217, 21, 0.007524980451609259],"Canvas":[43, 10, 0.09313943701763346],"KDD":[200, 45, 0.09769923007182914]}},
                    }

        print("trainging and testing  RSs")
        for baseline in ["EASE","UKNN","IKNN",'SVD','NMF','SLIM', 'IALS']: 
            print(baseline)
            train_rs = train_set.copy()
            best_CF_model = baseline
            if baseline == 'UKNN':
                cf_model = UserKNNCFRecommender(train_rs)
            elif baseline == 'IKNN':
                cf_model = ItemKNNCFRecommender(train_rs)
            elif baseline == 'IALS':
                cf_model = IALSRecommender(train_rs)
            elif baseline == 'SLIM':
                cf_model = SLIMElasticNetRecommender(train_rs)
            elif baseline == 'EASE':
                cf_model = EASE_R_Recommender(train_rs)
            elif baseline == 'NMF':
                cf_model = NMFRecommender(train_rs)
            elif baseline == 'SVD':
                cf_model = PureSVDRecommender(train_rs)

            
            best_param_list = scores[best_CF_model]['best_param'][dataset]
            best_param = dict(zip(scores[best_CF_model]["h_param"], best_param_list))
            cf_model.fit(**best_param)
            recom = recom_knn(cf_model,train_rs)

            l_list = [5,8,10]
            k_list = [3,5]
            log_print('Performance RE-RANKING')
            for k in k_list:
                for i in l_list:
                    log_print(f"k={k} Length list ={i} ", i)
                    log_print(f"{baseline}: ndcg of ",best_CF_model," ",ndcg(recom,test_set,k=k))
                    log_print("ndcg of ",f'{surv_model} on dropout'," ",ndcg(recomm_surv_1,test_set,k=k))
                    log_print("ndcg of ",f'{surv_model} on completion'," ",ndcg(recomm_surv_2,test_set,k=k))
                    log_print(" ndcg of ",f'{surv_model} on both'," ",ndcg(recomm_surv_3,test_set,k=k))
                    re_ranked_list1 = re_ranker(recom,recomm_surv_1,i,k)
                    re_ranked_list2 = re_ranker(recom,recomm_surv_2,i,k)
                    re_ranked_list3 = re_ranker(recom,recomm_surv_3,i,k)
                    log_print(f"{baseline} ndcg of re-ranking (base = recom and {surv_model} on dropout): ",    ndcg(re_ranked_list1,test_set,k=k))
                    log_print(f"{baseline} ndcg of re-ranking (base = recom and {surv_model} on completion): ",    ndcg(re_ranked_list2,test_set,k=k))
                    log_print(f"{baseline} ndcg of re-ranking (base = recom and {surv_model} on both): ",    ndcg(re_ranked_list3,test_set,k=k))
                    log_print(f"{baseline} ndcg-time of re-ranking (base = recom and {surv_model} on dropout): ",    ndcg_time(re_ranked_list1,test_set,test_time,k=k))
                    log_print(f"{baseline} ndcg-time of re-ranking (base = recom and {surv_model} on completion): ",    ndcg_time(re_ranked_list2,test_set,test_time,k=k))
                    log_print(f"{baseline} ndcg-time of re-ranking (base = recom and {surv_model} on both): ",    ndcg_time(re_ranked_list3,test_set,test_time,k=k))

                    tmp_res =  [surv_model,baseline,k,i,ndcg(recom,test_set,k=k),ndcg(recomm_surv_1,test_set,k=k),ndcg(recomm_surv_2,test_set,k=k),ndcg(recomm_surv_3,test_set,k=k),ndcg(re_ranked_list1,test_set,k=k),ndcg(re_ranked_list2,test_set,k=k),ndcg(re_ranked_list3,test_set,k=k),ndcg_time(re_ranked_list1,test_set,test_time,k=k),ndcg_time(re_ranked_list2,test_set,test_time,k=k),ndcg_time(re_ranked_list3,test_set,test_time,k=k)]
                    ranking_results.append(tmp_res)
    run_results = pd.DataFrame(ranking_results)
    run_results.columns = ['surv_model', 'baseline_model', 'k','list length','ndcg uknn','ndcg coxnet dropout','ndcg coxnet complete','ndcg coxnet both','ndcg re-rank dropout','ndcg re-rank completion','ndcg re-rank both','ndcg-time re-rank dropout','ndcg-time re-rank completion','ndcg-time re-rank both']
    return run_results

version = 'final_runs'
split_counts = [3] 
min_completeds = [1]
datasets = ['Canvas','X','KDD']
full_results = []
for dataset in datasets:
    logging.basicConfig(filename=f'{dataset}_experiments.txt', level=logging.INFO, format='%(message)s')
    for split_count in split_counts:
        for min_completed in min_completeds:
            if min_completed <= split_count:
                run_results = run_all_pca(dataset=dataset,split_count=split_count,min_completed=min_completed)
                run_results['split count'] = split_count
                run_results['min_completed'] = min_completed
                run_results['dataset'] = dataset
                full_results.append(run_results)
                log_print(run_results)
            else:
                pass
logging.basicConfig(filename=f'final_experiments_v{version}.txt', level=logging.INFO, format='%(message)s')
log_print('All results')
log_print(pd.concat(full_results))
pd.concat(full_results).to_csv(f'final_experiments_{version}.csv')


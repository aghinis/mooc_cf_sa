import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import os
import ast
from functools import partial
from pandasql import sqldf
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_density, stat_ecdf, scale_color_discrete, theme
#from baselines.IALSRecommender import IALSRecommender

# from skopt import forest_minimize
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



# Define a custom print function
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
        # data=pd.read_csv("~/data/dropRS/dataset/xuentangx_processed_v2.csv")
        # data=pd.read_csv("/home/u0111128/ml_codes/time_to_event/xuentangx_processed_v2.csv")
        data=pd.read_csv("xuentangx_processed_v2.csv")

        data["completed"] = data["completed"] + 1
        #data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.seconds/3600
        data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.total_seconds()/86400
    elif name == 'KDD':
        # data=pd.read_csv("~/data/dropRS/dataset/xuentangx_processed_v2.csv")
        # data=pd.read_csv("/home/u0111128/ml_codes/time_to_event/xuentangx_processed_v2.csv")
        data=pd.read_csv("kddcup_processed.csv")

        data["completed"] = data["completed"] + 1
        #data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.seconds/3600
        data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.total_seconds()/86400
    elif name == 'Canvas':
        cols_to_use = ['username','course_id','days_spent','completed']
        data=pd.read_csv("canvas_preprocessed.csv",index_col=False,usecols=cols_to_use)[cols_to_use]
        #data.loc[:, 'days_spent'] = pd.to_timedelta(data.loc[:, 'days_spent']).dt.total_seconds()/86400
    return data

def load_train_data(name):

    if name == 'kdd':
        df = pd.read_csv('../mooc_datasets/train_dfs/kdd_train_df.csv', low_memory=False)
        df.loc[:, 'days_spent'] = pd.to_timedelta(df.loc[:, 'days_spent']).dt.days

    elif name == 'xuen':
        df = pd.read_csv('~/data/dropRS/dataset/xuentangx_v2_train_df.csv', low_memory=False)
        df.loc[:, 'days_spent'] = pd.to_timedelta(df.loc[:, 'days_spent']).dt.seconds/3600

    elif name == 'xuen1':
        df = pd.read_csv('xuentangx_v1_train_df.csv', low_memory=False)
        df.loc[:, 'days_spent'] = pd.to_timedelta(df.loc[:, 'days_spent']).dt.total_seconds()/86400

    elif name == 'canvas':
        df = pd.read_csv('canvas_train_df.csv', low_memory=False)
        df.loc[:, 'days_spent'] = pd.to_timedelta(df.loc[:, 'days_spent']).dt.days
        df = df.drop(columns='Unnamed: 0')

    return df

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

def tune_and_fit_coxnet(dataset,X,y,event, tune=False):
    """
    Perform parameter tuning for Coxnet model and fit the final model.

    Parameters:
    X (DataFrame): The covariates.
    y (structured array): The structured array with fields 'Status' and 'Survival_in_days'.

    Returns:
    CoxnetSurvivalAnalysis: Fitted Coxnet model with the best parameters.
    """

    coxnet_params = {
        'completion' : {
            'X' : {'coxnetsurvivalanalysis__alphas': [0.0050]},
            'KDD' : {'coxnetsurvivalanalysis__alphas':[0.0639] },
            'Canvas': {'coxnetsurvivalanalysis__alphas':[0.236]},
        },
        'dropout' : {
             'X' : {'coxnetsurvivalanalysis__alphas':[0.0039] },
            'KDD' : {'coxnetsurvivalanalysis__alphas':[0.00941]} ,
            'Canvas': {'coxnetsurvivalanalysis__alphas':[0.335]},
        }
    }

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
    if tune == False:
        coxnet_pred.set_params(**coxnet_params[event][dataset])
    else:
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
    # unl_df = np.array([])
    # unl_df_names = np.array([])
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

def tune_rsf(dataset,X,y,event,tune=False):
    rsf_params = {
        'completion' : {
            'X' : {
                'n_estimators': 100,
                'max_features': 'sqrt',
                'min_samples_leaf' : 13,
                'min_samples_split' : 11,
                'max_depth': 12,
                'max_samples' : 0.5,
                'n_jobs' : 10
            },
            'KDD' : {
                'n_estimators': 100,
                'max_features': 'sqrt',
                'min_samples_leaf' : 12,
                'min_samples_split' : 18,
                'max_depth': 12,
                'max_samples' : 0.5,
                'n_jobs' : 10
            },
            'Canvas' : {
                'n_estimators': 80,
                'max_features': 'sqrt',
                'min_samples_leaf' : 18,
                'min_samples_split' : 20,
                'max_depth': 9,
                'max_samples' : 0.5,
                'n_jobs' : 10
            }
        },
        'dropout' : {
            'X' : {
            'n_estimators': 100,
            'max_features': 'sqrt',
            'min_samples_leaf' : 13,
            'min_samples_split' : 11,
            'max_depth': 12,
            'max_samples' : 0.5,
            'n_jobs' : 10
        },
        'KDD' : {
            'n_estimators': 100,
            'max_features': 'sqrt',
            'min_samples_leaf' : 12,
            'min_samples_split' : 18,
            'max_depth': 12,
            'max_samples' : 0.5,
            'n_jobs' : 10
        },
        'Canvas' : {
            'n_estimators': 80,
            'max_features': 'sqrt',
            'min_samples_leaf' : 18,
            'min_samples_split' : 20,
            'max_depth': 9,
            'max_samples' : 0.5,
            'n_jobs' : 10
        }
        }
    }
    fname = f'trained_models/{dataset}_{event}_rsf.pkl'
    if tune==False:
        model = RandomSurvivalForest(**rsf_params[event][dataset]).fit(X,y)
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

def tune_boosted(dataset,X,y,event, tune=False):
    fname  = f'trained_models/{dataset}_{event}_xgb.pkl'
    xgb_params = {
        'completion' : {
            'X' : {
                'learning_rate': 0.3835277306131044,
                'n_estimators': 173,
                'max_features': 'sqrt',
                'min_samples_leaf' : 18,
                'min_samples_split' : 13,
                'max_depth': 5,
                'subsample' : 0.5 
            },
            'Canvas' : {
                'learning_rate': 0.11176503359259893,
                'n_estimators': 153,
                'max_features': 'sqrt',
                'min_samples_leaf' : 10,
                'min_samples_split' : 12,
                'max_depth': 10,
                'subsample' : 0.5 

            },
            'KDD' : {
                'learning_rate': 0.1495643837287974,
                'n_estimators': 124,
                'max_features': 'sqrt',
                'min_samples_leaf' : 8,
                'min_samples_split' : 12,
                'max_depth': 3,
                'subsample' : 0.5 
            }
            },
        'dropout' : {
                'X' : {
                'learning_rate': 0.2770816286226958,
                'n_estimators': 184,
                'max_features': 'sqrt',
                'min_samples_leaf' : 16,
                'min_samples_split' : 6,
                'max_depth': 7,
                'subsample' : 0.5 
            },
            'Canvas' : {
                'learning_rate': 0.12471647045579379,
                'n_estimators': 199,
                'max_features': 'sqrt',
                'min_samples_leaf' : 15,
                'min_samples_split' : 17,
                'max_depth': 12,
                'subsample' : 0.5 

            },
            'KDD' : {
                'learning_rate': 0.41242104750758624,
                'n_estimators': 121,
                'max_features': 'sqrt',
                'min_samples_leaf' : 16,
                'min_samples_split' : 6,
                'max_depth': 16,
                'subsample' : 0.5 
            }
            }
        }
    if tune ==False:
        model = GradientBoostingSurvivalAnalysis(**xgb_params[event][dataset]).fit(X,y)
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


course_mapping_dict = {
    'course-v1:FUDANx+ECON130007_01+2016_T1' : 'course-v1:FUDANx+ECON130007_01+2016_T2',
    'course-v1:MITx+15_390x_2015_T2+2015_T2' : 'course-v1:MITx+15_390x_2016T1+2016_T1',
    'course-v1:MITx+15_390x_2016T1+2016_TS' : 'course-v1:MITx+15_390x_2016T1+2016_T1',
    'course-v1:MITx+6_00_1x+2015_T1' : 'course-v1:MITx+6_00_1x+2015_T2',
    'course-v1:TsinghuaX+00040132X+2016_T1' : 'course-v1:TsinghuaX+00040132X+2016_TS',
    'course-v1:TsinghuaX+00510663X+2015_T2' : 'course-v1:TsinghuaX+00510663X+2016_T1', 
    'course-v1:TsinghuaX+00510663X+2016_T2' : 'course-v1:TsinghuaX+00510663X+2016_T1', 
    'course-v1:TsinghuaX+00510663X+2016_TS' : 'course-v1:TsinghuaX+00510663X+2016_T1', 
    'course-v1:TsinghuaX+00510888X+2016_T1' : 'course-v1:TsinghuaX+00510888X_2015_T2+2015_T2',
    'course-v1:TsinghuaX+00612642X+2016_T1' : 'course-v1:TsinghuaX+00612642X+2016_T2',
    'course-v1:TsinghuaX+00612642X+2016_TS' : 'course-v1:TsinghuaX+00612642X+2016_T2',
    'course-v1:TsinghuaX+00612642X_2015_2+2015_T2' : 'course-v1:TsinghuaX+00612642X+2016_T2',
    'course-v1:TsinghuaX+00680082_1X+2016_T1' : 'course-v1:TsinghuaX+00680082_1X+2016_T2',
    'course-v1:TsinghuaX+00690092X+2016_T1' : 'course-v1:TsinghuaX+00690092X+2016_T2',
    'course-v1:TsinghuaX+00690092X+2017_T1': 'course-v1:TsinghuaX+00690092X+2016_T2',
    'course-v1:TsinghuaX+00690092X_2015_T2+2015_T2': 'course-v1:TsinghuaX+00690092X+2016_T2',
    'course-v1:TsinghuaX+00690212X+2015_T2':  'course-v1:TsinghuaX+00690212X+2016_T1',
    'course-v1:TsinghuaX+00690342X+2016_T1' : 'course-v1:TsinghuaX+00690342X+2016_T2', 
    'course-v1:TsinghuaX+00690342X+2016_TS': 'course-v1:TsinghuaX+00690342X+2016_T2', 
    'course-v1:TsinghuaX+00690342X+2017_T1': 'course-v1:TsinghuaX+00690342X+2016_T2', 
    'course-v1:TsinghuaX+00690863X+2016_TS' : 'course-v1:TsinghuaX+00690863X+2017_T1',
    'course-v1:TsinghuaX+00690863X+__': 'course-v1:TsinghuaX+00690863X+2017_T1',
    'course-v1:TsinghuaX+00691153X+2016_T1' : 'course-v1:TsinghuaX+00691153X+2016_T2',
    'course-v1:TsinghuaX+00740043X+2016_T1' : 'course-v1:TsinghuaX+00740043X+2016_TS',
    'course-v1:TsinghuaX+00740043X_2015_T2+2015_T2' : 'course-v1:TsinghuaX+00740043X+2016_TS',
    'course-v1:TsinghuaX+00740043_1x+2017_T1': 'course-v1:TsinghuaX+00740043X+2016_TS',
    'course-v1:TsinghuaX+00740043_2X+2016_T1' : 'course-v1:TsinghuaX+00740043X+2016_TS',
    'course-v1:TsinghuaX+00740123X+2016_T1' : 'course-v1:TsinghuaX+00740123X+2016_TS', 
    'course-v1:TsinghuaX+00740123X+2017_T1' : 'course-v1:TsinghuaX+00740123X+2016_TS',
    'course-v1:TsinghuaX+10610183X_2015_T2+2015_T2' : 'course-v1:TsinghuaX+10610183_2X+2016_T1',
    'course-v1:TsinghuaX+10610183_2X+2016_T2' : 'course-v1:TsinghuaX+10610183_2X+2016_T1',
    'course-v1:TsinghuaX+10610193X+2015_T3' : 'course-v1:TsinghuaX+10610193X+2016_T1',
    'course-v1:TsinghuaX+10610193X+2017_T1' : 'course-v1:TsinghuaX+10610193X+2016_T1',
    'course-v1:TsinghuaX+10610224X+2015_T3' : 'course-v1:TsinghuaX+10610224X+2016-T2',
    'course-v1:TsinghuaX+10610224X+2016_T1' : 'course-v1:TsinghuaX+10610224X+2016-T2',
    'course-v1:TsinghuaX+10610224X+2017_T1' : 'course-v1:TsinghuaX+10610224X+2016-T2',
    'course-v1:TsinghuaX+10620204X+2016_T1' : 'course-v1:TsinghuaX+10620204X+2016_T2', 
    'course-v1:TsinghuaX+10620204X+2017_T1' : 'course-v1:TsinghuaX+10620204X+2016_T2',
    'course-v1:TsinghuaX+20220214X+2016_T1' :  'course-v1:TsinghuaX+20220214X+2016_T2',
    'course-v1:TsinghuaX+20220214X+2017_T1' : 'course-v1:TsinghuaX+20220214X+2016_T2',
    'course-v1:TsinghuaX+20220214X_2015_2+2015_T2' : 'course-v1:TsinghuaX+20220214X+2016_T2',
    'course-v1:TsinghuaX+20220332X+2016_T1' : 'course-v1:TsinghuaX+20220332X+2017_T1',
    'course-v1:TsinghuaX+20250064+2015_T2' :  'course-v1:TsinghuaX+20250064+2016_T2',
    'course-v1:TsinghuaX+20250064_1X+2015_T2' : 'course-v1:TsinghuaX+20250064+2016_T2',
    'course-v1:TsinghuaX+20250103X+2016--T2' : 'course-v1:TsinghuaX+20250103X+2016_T1', 
    'course-v1:TsinghuaX+20250103X_+2017_T1' : 'course-v1:TsinghuaX+20250103X+2016_T1',
    'course-v1:TsinghuaX+20440333X+2016_T1' : 'course-v1:TsinghuaX+20440333_2015X+2015_T2',
    'course-v1:TsinghuaX+30240184+2015_T2' : 'course-v1:TsinghuaX+30240184_1X+2016_T1',
    'course-v1:TsinghuaX+30240184_1X+2016_TS' : 'course-v1:TsinghuaX+30240184_1X+2016_T1',
    'course-v1:TsinghuaX+30240184_2X+2016_T1' : 'course-v1:TsinghuaX+30240184_1X+2016_T1',
    'course-v1:TsinghuaX+30640014+2015_T2' : 'course-v1:TsinghuaX+30640014+2016_T2' , 
    'course-v1:TsinghuaX+30640014X+2016_T1' : 'course-v1:TsinghuaX+30640014+2016_T2',
    'course-v1:TsinghuaX+30640014X+2016_T2' : 'course-v1:TsinghuaX+30640014+2016_T2',
    'course-v1:TsinghuaX+30640014X+2017_T1' : 'course-v1:TsinghuaX+30640014+2016_T2',
    'course-v1:TsinghuaX+30700313X+2016_T1' : 'course-v1:TsinghuaX+30700313X+2016_T2',
    'course-v1:TsinghuaX+30700313X+2016_TS' : 'course-v1:TsinghuaX+30700313X+2016_T2',
    'course-v1:TsinghuaX+30700313X+2017_T1' : 'course-v1:TsinghuaX+30700313X+2016_T2',
    'course-v1:TsinghuaX+34000888X+2016_T1' : 'course-v1:TsinghuaX+34000888X+2016_TS', 
    'course-v1:TsinghuaX+34100325X+2016_T1' : 'course-v1:TsinghuaX+34100325X+sp',
    'course-v1:TsinghuaX+40670453X+2016_T1' : 'course-v1:TsinghuaX+40670453X+2016_T2',
    'course-v1:TsinghuaX+40670453X+2016_TS' : 'course-v1:TsinghuaX+40670453X+2016_T2',
    'course-v1:TsinghuaX+64100033X+2015_T1_' : 'course-v1:TsinghuaX+64100033X+2016_TS',
    'course-v1:TsinghuaX+70800232X+2015_T2' : 'course-v1:TsinghuaX+70800232X+2016_T1',
    'course-v1:TsinghuaX+80000901X+2016-T2' : 'course-v1:TsinghuaX+80000901X+2016_T1', 
    'course-v1:TsinghuaX+80000901X+2016_TS1' : 'course-v1:TsinghuaX+80000901X+2016_T1', 
    'course-v1:TsinghuaX+80000901X_1+2017_T1': 'course-v1:TsinghuaX+80000901X+2016_T1', 
    'course-v1:TsinghuaX+80000901X_2015T2+2015T2': 'course-v1:TsinghuaX+80000901X+2016_T1', 
    'course-v1:TsinghuaX+80511503X+2016_T1' : 'course-v1:TsinghuaX+80511503X+2016_T2', 
    'course-v1:TsinghuaX+80511503X+2017_T1': 'course-v1:TsinghuaX+80511503X+2016_T2',
    'course-v1:TsinghuaX+80512073X+2016--T2' : 'course-v1:TsinghuaX+80512073X+2016_T1',
    'course-v1:TsinghuaX+80512073X+2016_TS' : 'course-v1:TsinghuaX+80512073X+2016_T1',
    'course-v1:TsinghuaX+80512073X+2017-T1' : 'course-v1:TsinghuaX+80512073X+2016_T1',
    'course-v1:TsinghuaX+80512073X_2015_2+2015_T2' : 'course-v1:TsinghuaX+80512073X+2016_T1',
    'course-v1:TsinghuaX+90640012X+2016-T2': 'course-v1:TsinghuaX+90640012X+2016_T1', 
    'course-v1:TsinghuaX+90640012X+2017_T1' : 'course-v1:TsinghuaX+90640012X+2016_T1',
    'course-v1:TsinghuaX+AP000002X+2016_T1' : 'course-v1:TsinghuaX+AP000002X+2016_T2',
    'course-v1:TsinghuaX+AP000003X+2016_T1' : 'course-v1:TsinghuaX+AP000003X+2016_T2',
    'course-v1:TsinghuaX+AP000004X+2016-2' : 'course-v1:TsinghuaX+AP000004X+2016_T1', 
    'course-v1:TsinghuaX+AP000004X+2016_T2' : 'course-v1:TsinghuaX+AP000004X+2016_T1', 
    'course-v1:TsinghuaX+AP000005X+2016-2' : 'course-v1:TsinghuaX+AP000005X+2016_T1', 
    'course-v1:TsinghuaX+AP000005X+2016_T2': 'course-v1:TsinghuaX+AP000005X+2016_T1',
    'course-v1:TsinghuaX+JRFX01+2016--T2' : 'course-v1:TsinghuaX+JRFX01+2016_TS',
    'course-v1:TsinghuaX+JRFX01+2017_T1': 'course-v1:TsinghuaX+JRFX01+2016_TS', 
    'course-v1:TsinghuaX+THU00001X+2016_T1' : 'course-v1:TsinghuaX+THU00001X+2016_T2',
    'course-v1:UC_BerkeleyX+ColWri2_1x+2015_T2' : 'course-v1:UC_BerkeleyX+ColWri2_1x_2015_T1+2016_TS'

}


def run_all_pca(dataset,split_count=3,min_completed=1, normalize_time=True, tune_models=False):
    log_print(f"\n Starting: {dataset} split_count={split_count}, min_completed = {min_completed}, normalized_time={normalize_time}, PCA on intercations")
    num_folds_tunning = 3
    tunning = False
    njobs = -1
    random_seed = np.random.randint(100)
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

    if dataset == 'Canvas':
        data = data[data['username']!=832988535]

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
    #df = df.sample(frac=0.05, replace=False, random_state=2)
    bin_mtx, time_mtx = df_to_matrix(df)
    u_b_int, u_t_int, c_b_int, c_t_int = get_interaction_feats(bin_mtx, time_mtx)
    u_b_int.columns = [f'{col}_u_b' for col in u_b_int.columns]
    u_t_int.columns = [f'{col}_u_t' for col in u_t_int.columns]
    c_b_int.columns = [f'{col}_c_b' for col in c_b_int.columns]
    c_t_int.columns = [f'{col}_c_t' for col in c_t_int.columns]

    df2 = min_max_normalize(df,['course_id','completed'],'days_spent')
    #df2 = df2.sample(frac=0.05, replace=False, random_state=2)
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
        df_with_feats1 = df2.loc[:, ['username', 'course_id', 'days_spent', 'completed']].join(u_b_int, on='username', how='left')#, rsuffix='_u_b')
    else:
        df_with_feats1 = df.loc[:, ['username', 'course_id', 'days_spent', 'completed']].join(u_b_int, on='username', how='left')#, rsuffix='_u_b')
    df_with_feats1 = df_with_feats1.join(u_t_int, on='username', how='left')# rsuffix='_u_t')
    df_with_feats1 = df_with_feats1.join(c_b_int, on='course_id', how='left')# rsuffix='_c_b')
    df_with_feats1 = df_with_feats1.join(c_t_int, on='course_id', how='left')# rsuffix='_c_t')
    if dataset=='X':
        df_with_feats1 = df_with_feats1.merge(feature_course, on='course_id')# how='left')
    df_with_feats1 = df_with_feats1.merge(user_features, on='username')# how='left')

    X_train = df_with_feats1.drop(columns=['username', 'course_id', 'days_spent', 'completed'])
    y_train = df_with_feats1.loc[:, ['days_spent', 'completed']]
    X_train.head()

    pca_pipeline_1 = Pipeline([
        ('scaler', StandardScaler()),  # Optional: standardize the data before PCA
        ('pca', PCA(n_components=0.8, svd_solver = 'full'))
    ])        
    # Combine transformers with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('pca1', pca_pipeline_1, u_b_int.columns.tolist()+u_t_int.columns.tolist()+ c_b_int.columns.tolist()+c_t_int.columns.tolist()),
            ('user_scaler', StandardScaler(), user_features.columns.tolist()),
        #  ('education_dummies', OneHotEncoder(drop='first'), ['education']),
        # ('sex_dummy', OneHotEncoder(drop='first'), ['gender']) 
        ]
    )
    transformed_data = preprocessor.fit_transform(X_train)
    pca1_names = [f'PCA_bin{i+1}' for i in range(len(preprocessor.named_transformers_['pca1'].named_steps['pca'].explained_variance_))]
    final_column_names = pca1_names  + user_features.columns.tolist() #+ pca3_names + age_name# + list(education_names) + list(sex_names)

    transformed_df = pd.DataFrame(transformed_data, columns=final_column_names)
    # define dropout
    y_test_dropout = [(True,t[0]) if t[1]==1 else (False,t[0]) for t in y_train.values]
    y_test_dropout = np.array(y_test_dropout, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    # define completion
    y_test_completion = [(True,t[0]) if t[1]==2 else (False,t[0]) for t in y_train.values]
    y_test_completion = np.array(y_test_completion, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    ranking_results = list()
    #surv_models = ['Coxnet','rsf','XGb']
    surv_models = ['Coxnet']
    for surv_model in surv_models: 
        log_print(f'Starting {surv_model}')
        if surv_model == 'Coxnet': 
            print("training dropout")
            model_dropout = tune_and_fit_coxnet(dataset,transformed_df, y_test_dropout,'dropout',tune=tune_models)
            print("Training Completion")
            model_completion = tune_and_fit_coxnet(dataset,transformed_df, y_test_completion,'completion',tune=tune_models)
        elif surv_model =='rsf':
            print("training dropout")
            model_dropout = tune_rsf(dataset,transformed_df, y_test_dropout,'dropout',tune=tune_models)
            print("Training Completion")
            model_completion = tune_rsf(dataset,transformed_df, y_test_completion, 'completion',tune=tune_models)
        elif surv_model == 'XGb':
            print("training dropout")
            model_dropout = tune_boosted(dataset,transformed_df, y_test_dropout,'dropout',tune=tune_models)
            print("Training Completion")
            model_completion = tune_boosted(dataset,transformed_df, y_test_completion,'completion',tune=tune_models)
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        cv_fold_dropout = cross_val_score(model_dropout,transformed_df, y_test_dropout,cv=cv, n_jobs=-1)
        cv_fold_completion = cross_val_score(model_completion,transformed_df, y_test_completion,cv=cv, n_jobs=-1)
        log_print(f'Dropout C-index {surv_model}: {cv_fold_dropout.mean()} with {cv_fold_dropout}')
        log_print(f'Completion C-index {surv_model}: {cv_fold_completion.mean()} with {cv_fold_completion}')
        c_index_results = pd.DataFrame({'dataset': dataset,
                                        'model' : surv_model,
                                        'c_index_dropout':cv_fold_dropout,
                                        'c_index_completion':cv_fold_completion})
        stacked_bin_mtx = bin_mtx.where(bin_mtx == 0).stack(level=0)
        batch_size = 10000
        num_rows = len(stacked_bin_mtx)
        num_batches = (num_rows + batch_size - 1) // batch_size  # Calculate the number of batches

        def process_batch(batch):
            num_rows = len(batch)
            num_cols = X_train.shape[1] # u_b_int.shape[1] + u_t_int.shape[1] + c_b_int.shape[1] + c_t_int.shape[1] + kk.shape[1]
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

            ######################################
        ##### PART 1 - Train, Tune and test RSs #######
        ######################################
        scores = {
                    # "BPR":{"ndcg":[],"MAP":[],"recall":[],"precision":[],"h_param":['epochs','no_components','learning_rate','item_alpha','user_alpha'],"h_param_range":[(50, 350),(10,300),(10**-5, 10**-1, 'log-uniform'),(10**-7, 10**-1, 'log-uniform'),(10**-6, 10**-1, 'log-uniform')],'best_param':[336, 207, 5.596974933016647e-05, 0.0005605913885321211, 0.07893887705085302]},
                # "WARP":{"ndcg":[],"MAP":[],"recall":[],"precision":[],"h_param":['epochs','no_components','learning_rate','item_alpha','user_alpha'],"h_param_range":[(50, 350),(10,300),(10**-5, 10**-1, 'log-uniform'),(10**-7, 10**-1, 'log-uniform'),(10**-6, 10**-1, 'log-uniform')],'best_param':[268, 226, 7.01854105327644e-05, 0.006705265022562404, 0.01602820393938058]},
                # "MVAE":{"ndcg":[],"h_param":['epochs','batch_size','total_anneal_steps'],"h_param_range":[(10,250),(25,500),(100000,300000)],'best_param':[99, 391, 127354]},
                "EASE":{"ndcg":[],"h_param":['topK', 'l2_norm'],"h_param_range":[[None],(1e0, 1e7)],'best_param':{"X":[None, 95109.6942962282],"Canvas":[None, 9325573.660829231],"KDD":[None, 2540.6584595004156]}},
                "UKNN":{"ndcg":[],"h_param":['topK', 'shrink'],"h_param_range":[(20, 800),(0,1000)],'best_param':{"X":[301, 178],"Canvas":[128, 8],"KDD":[488, 907]}},
                "IKNN":{"ndcg":[],"h_param":['topK', 'shrink'],"h_param_range":[(20, 800),(0,1000)],'best_param':{"X":[70, 350],"Canvas":[789, 793],"KDD":[37, 194]}},
                "SVD":{"ndcg":[],"h_param":['num_factors','random_seed'],"h_param_range":[(3, 300),[int(random_seed)]],'best_param':{"X":[5, 1],"Canvas":[8, 1],"KDD":[3, 1]}},
                "NMF":{"ndcg":[],"h_param":['num_factors','l1_ratio'],"h_param_range":[(10, 300),(0.1,0.9)],'best_param':{"X":[202, 0.21397251366409453],"Canvas":[242, 0.2322551413506923],"KDD":[43, 0.8460852461396278]}},
                "SLIM":{"ndcg":[],"h_param":['topK', 'l1_ratio','alpha'],"h_param_range":[(5, 600),(1e-5,1.0),(1e-3, 1.0)],'best_param': {"X":[380, 0.00023022491889188525, 0.31065433709916135],"Canvas":[486, 0.5934988381321606, 0.00590093297649861],"KDD":[321, 0.03911733924386367, 0.17916129385389876]}},
                "IALS":{"ndcg":[],"h_param":['epochs','num_factors','reg'],"h_param_range":[(10, 200),(10,100),(1e-5, 1e-1)],'best_param':{"X":[217, 21, 0.007524980451609259],"Canvas":[43, 10, 0.09313943701763346],"KDD":[200, 45, 0.09769923007182914]}},
                    }

        print("trainging and testing  RSs")
        #
        train_rs = train_set.copy()
        # for row, col in zip(*train_rs.nonzero()):
        #     if train_rs[row, col] == 2:
        #         train_rs[row, col] = 1
        best_CF_model = "UKNN"
        cf_model = UserKNNCFRecommender(train_rs)
        best_param_list = scores[best_CF_model]['best_param'][dataset]
        best_param = dict(zip(scores[best_CF_model]["h_param"], best_param_list))
        cf_model.fit(**best_param)
        recom = recom_knn(cf_model,train_rs)


        print("Performance BASE MODEL")
        print("ndcg of ",best_CF_model," ",ndcg(recom,test_set,k=3))

        print("Performance COXNet")
        print("ndcg of ",f'{surv_model}'," ",ndcg(recomm_surv_1,test_set,k=3))
    
        l_list = [5,8,10]
        k_list = [3,5]
        log_print('Performance RE-RANKING')
        for k in k_list:
            for i in l_list:
                log_print(f"k={k} Length list ={i} ", i)
                log_print("ndcg of ",best_CF_model," ",ndcg(recom,test_set,k=k))
                log_print("ndcg of ",f'{surv_model} on dropout'," ",ndcg(recomm_surv_1,test_set,k=k))
                log_print("ndcg of ",f'{surv_model} on completion'," ",ndcg(recomm_surv_2,test_set,k=k))
                log_print("ndcg of ",f'{surv_model} on both'," ",ndcg(recomm_surv_3,test_set,k=k))
                re_ranked_list1 = re_ranker(recom,recomm_surv_1,i,k)
                re_ranked_list2 = re_ranker(recom,recomm_surv_2,i,k)
                re_ranked_list3 = re_ranker(recom,recomm_surv_3,i,k)
                log_print(f"ndcg of re-ranking (base = recom and {surv_model} on dropout): ",    ndcg(re_ranked_list1,test_set,k=k))
                log_print(f"ndcg of re-ranking (base = recom and {surv_model} on completion): ",    ndcg(re_ranked_list2,test_set,k=k))
                log_print(f"ndcg of re-ranking (base = recom and {surv_model} on both): ",    ndcg(re_ranked_list3,test_set,k=k))
                log_print(f"ndcg-time of re-ranking (base = recom and {surv_model} on dropout): ",    ndcg_time(re_ranked_list1,test_set,test_time,k=k))
                log_print(f"ndcg-time of re-ranking (base = recom and {surv_model} on completion): ",    ndcg_time(re_ranked_list2,test_set,test_time,k=k))
                log_print(f"ndcg-time of re-ranking (base = recom and {surv_model} on both): ",    ndcg_time(re_ranked_list3,test_set,test_time,k=k))

                tmp_res =  [surv_model,k,i,ndcg(recom,test_set,k=k),ndcg(recomm_surv_1,test_set,k=k),ndcg(recomm_surv_2,test_set,k=k),ndcg(recomm_surv_3,test_set,k=k),ndcg(re_ranked_list1,test_set,k=k),ndcg(re_ranked_list2,test_set,k=k),ndcg(re_ranked_list3,test_set,k=k),ndcg_time(re_ranked_list1,test_set,test_time,k=k),ndcg_time(re_ranked_list2,test_set,test_time,k=k),ndcg_time(re_ranked_list3,test_set,test_time,k=k)]
                # if dataset != 'Canvas':
                #     re_ranked_list2 = re_ranker(recomm_surv_1,recom,i,k)
                #     log_print("ndcg of re-ranking (base = surv): ",    ndcg(re_ranked_list2,test_set,k=k))
                ranking_results.append(tmp_res)
    run_results = pd.DataFrame(ranking_results)
    run_results.columns = ['surv_model', 'k','list length','ndcg uknn','ndcg coxnet dropout','ndcg coxnet complete','ndcg coxnet both','ndcg re-rank dropout','ndcg re-rank completion','ndcg re-rank both','ndcg-time re-rank dropout','ndcg-time re-rank completion','ndcg-time re-rank both']
    return run_results, c_index_results

version = 'revision_runs'
split_counts = [3] 
min_completeds = [1]
#datasets = ['Canvas','X','KDD']
datasets = ['Canvas']
full_results = []
c_index_results = []
for dataset in datasets:
    logging.basicConfig(filename=f'{dataset}_experiments_v{version}.txt', level=logging.INFO, format='%(message)s')
    for split_count in split_counts:
        for min_completed in min_completeds:
            if min_completed <= split_count:
                for iteration in range(0,2):
                    print(iteration)
                    all_results = run_all_pca(dataset=dataset,split_count=split_count,min_completed=min_completed)
                    run_results = all_results[0]
                    run_results['split count'] = split_count
                    run_results['min_completed'] = min_completed
                    run_results['dataset'] = dataset
                    run_results['iteration'] = iteration
                    full_results.append(run_results)
                    log_print(run_results)
                    c_index_run = all_results[1]
                    c_index_run['split count'] = split_count
                    c_index_run['min_completed'] = min_completed
                    c_index_run['dataset'] = dataset
                    c_index_run['iteration'] = iteration
                    c_index_results.append(c_index_run)
logging.basicConfig(filename=f'trial_new{version}.txt', level=logging.INFO, format='%(message)s')
log_print('All results')
log_print(pd.concat(full_results))
pd.concat(full_results).to_csv(f'nn_{version}.csv')
pd.concat(c_index_results).to_csv(f'c_index_{version}.csv')

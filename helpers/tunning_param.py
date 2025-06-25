import numpy as np
from skopt import forest_minimize
# from baselines.IALSRecommender import IALSRecommender
# import lightfm
from helpers.measures import ndcg
# from helpers.utils import recom_lf, recom_knn
from helpers.utils import recom_knn,recom_librec,matrix_to_df,prep_data_librec

from baselines.knn import ItemKNNCFRecommender,UserKNNCFRecommender
from baselines.PureSVDRecommender import PureSVDRecommender
from baselines.NMFRecommender import NMFRecommender
from baselines.EASE_R_Recommender import EASE_R_Recommender
from baselines.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from baselines.IALSRecommender import IALSRecommender

from libreco.algorithms import LightGCN,NGCF
from libreco.data import DatasetPure

def objective_ngcf(params, train, validation,rid_idx, cid_idx,idx_to_rid, idx_to_cid):
    embed, epoch,l_r ,reg_ = params
   
    train_df = matrix_to_df(train,idx_to_rid, idx_to_cid)
    eval_df = matrix_to_df(validation,idx_to_rid, idx_to_cid)
    train_df = prep_data_librec(train_df)
    eval_df = prep_data_librec(eval_df)
    train_data, data_info = DatasetPure.build_trainset(train_df)
    eval_data = DatasetPure.build_trainset(eval_df)
    

    model = NGCF(
            "ranking",
            data_info,
            loss_type="cross_entropy",
            embed_size=embed,
            n_epochs=epoch,
            lr=l_r,
            lr_decay=False,
            reg=reg_,
            batch_size=2048,
            num_neg=1,
            node_dropout=0.0,
            message_dropout=0.0,
            hidden_units=(32, 32, 32),
            device="cuda",
        )
 
    model.fit(
                train_data,
                neg_sampling=True,
                verbose=0,
                shuffle=True,
                eval_data=eval_data,
                metrics=[
                "loss",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "ndcg",
                ],
                )    
    
    recoms = recom_librec(model,train_df,train,rid_idx,cid_idx)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out


def objective_lightgcn(params, train, validation,rid_idx, cid_idx,idx_to_rid, idx_to_cid):
    embed, epoch,l_r ,reg_ ,n_layer = params
   
    train_df = matrix_to_df(train,idx_to_rid, idx_to_cid)
    eval_df = matrix_to_df(validation,idx_to_rid, idx_to_cid)
    train_df = prep_data_librec(train_df)
    eval_df = prep_data_librec(eval_df)
    train_data, data_info = DatasetPure.build_trainset(train_df)
    eval_data = DatasetPure.build_trainset(eval_df)
    

    model = LightGCN(
                    "ranking",
                    data_info,
                    loss_type="bpr",
                    embed_size=embed,
                    n_epochs=epoch,
                    lr=l_r,
                    lr_decay=False,
                    reg=reg_,
                    batch_size=2048,
                    num_neg=1,
                    dropout_rate=0.0,
                    n_layers=n_layer,
                    device="cuda",
                    )
 
    model.fit(
                train_data,
                neg_sampling=True,
                verbose=0,
                shuffle=True,
                eval_data=eval_data,
                metrics=[
                "loss",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "ndcg",
                ],
                )    
    
    recoms = recom_librec(model,train_df,train,rid_idx,cid_idx)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_IALS(params, train, validation):
    iterations, factors, regu = params
    model = IALSRecommender(train)
    model.fit(num_factors=factors,epochs=iterations,reg=regu)
    recoms = recom_knn(model,train)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_IKNN(params, train, validation):
    neighbors,shrink = params #ALS
    model = ItemKNNCFRecommender(train)
    model.fit(topK=neighbors, shrink=shrink, similarity='cosine', normalize=True, feature_weighting = "TF-IDF")
    recoms = recom_knn(model,train)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_UKNN(params, train, validation):
    neighbors,shrink = params
    model = UserKNNCFRecommender(train)
    model.fit(topK=neighbors, shrink=shrink, similarity='cosine', normalize=True, feature_weighting = "TF-IDF")
    recoms = recom_knn(model,train)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out


def objective_EASE(params,train, validation):
    factor,l2 = params
    model = EASE_R_Recommender(train)
    model.fit(topK =factor,  l2_norm = l2)
    recoms = recom_knn(model,train)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_SVD(params,train, validation):
    factor,l2 = params
    model = PureSVDRecommender(train)
    model.fit(num_factors=factor)
    out = - ndcg(train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_NMF(params,train, validation):
    factor,l2 = params
    model = NMFRecommender(train)
    model.fit(num_factors=factor,l1_ratio = l2)
    out = - ndcg(train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_SLIM(params,train, validation):
    topK,l1_ratio,alpha = params
    model = SLIMElasticNetRecommender(train)
    model.fit(topK = topK, l1_ratio = l1_ratio, alpha =alpha)
    out = - ndcg(train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def tune(objective,hyperparam,n_calls,njobs,random_seed):
    search_space = hyperparam
    results = forest_minimize(objective, search_space, n_calls=40,n_jobs=njobs,random_state=random_seed,verbose=True)
    return results
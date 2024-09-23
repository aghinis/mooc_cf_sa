import numpy as np
from skopt import forest_minimize
from baselines.knn import ItemKNNCFRecommender,UserKNNCFRecommender
# from baselines.IALSRecommender import IALSRecommender
# import lightfm
from helpers.measures import ndcg
# from helpers.utils import recom_lf, recom_knn
from helpers.utils import recom_knn

# from PureSVDRecommender import PureSVDRecommender
# from NeuMF_our_interface.NeuMF_RecommenderWrapper import NeuMF_RecommenderWrapper
# from NMFRecommender import NMFRecommender
# from baselines.EASE_R_Recommender import EASE_R_Recommender
# from MultiVAE_our_interface.MultiVAE_RecommenderWrapper import Mult_VAE_RecommenderWrapper
# from SLIMElasticNetRecommender import SLIMElasticNetRecommender


def objective_BPR(params, train, validation):
    iterations, factors, u_regu, i_regu, lr = params
    model = lightfm.LightFM(loss='bpr',
                    learning_rate=lr,
                    no_components=factors,
                    user_alpha=u_regu,
                    item_alpha=i_regu,
                    random_state=0)
    model.fit(train,epochs=iterations, verbose=True)
    recoms = recom_lf(model,train)
    out = - ndcg(recoms,validation,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_WARP(params, train, validation):
    iterations,factors, u_regu, i_regu, lr = params
    model = lightfm.LightFM(loss='warp',
                    learning_rate=lr,
                    no_components=factors,
                    user_alpha=u_regu,
                    item_alpha=i_regu,
                    random_state=0)
    model.fit(train,epochs=iterations, verbose=True)
    recoms = recom_lf(model,train)
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
def objective_MVAE(params, train, validation):
    iterations,b_size,aneal = params
    model = Mult_VAE_RecommenderWrapper(train)
    # for i in range(len(p_dim)):
    #     p_dim[i].append(model.n_items)
    model.fit(epochs=iterations,batch_size=b_size,total_anneal_steps=aneal)
    out = - ndcg_knn (train,validation,model,k=10)
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

def tune(objective,hyperparam,n_calls,njobs,random_seed):
    search_space = hyperparam
    results = forest_minimize(objective, search_space, n_calls=40,n_jobs=njobs,random_state=random_seed,verbose=True)
    return results

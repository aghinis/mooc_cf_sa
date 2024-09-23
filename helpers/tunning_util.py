from skopt import forest_minimize
from knn import ItemKNNCFRecommender,UserKNNCFRecommender
from PureSVDRecommender import PureSVDRecommender
from NeuMF_our_interface.NeuMF_RecommenderWrapper import NeuMF_RecommenderWrapper
from NMFRecommender import NMFRecommender
from EASE_R_Recommender import EASE_R_Recommender
# from MultiVAE_our_interface.MultiVAE_RecommenderWrapper import Mult_VAE_RecommenderWrapper
from SLIMElasticNetRecommender import SLIMElasticNetRecommender
from P3alphaRecommender import P3alphaRecommender
from RP3betaRecommender import RP3betaRecommender
from measures import MAP,MAP_knn, MAP_lf, precision, precision_knn, precision_lf, recall, recall_knn, recall_lf,ndcg, ndcg_knn, ndcg_lf, precision_nrlmf,recall_nrlmf,MAP_nrlmf,ndcg_nrlmf, ndcg_cold_item, ndcg_tf,ndcg_cold_cmf


def objective_svd(params):
    factor,l2 = params
    model = PureSVDRecommender(train)
    model.fit(num_factors=factor)
    out = - ndcg_knn (train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_NMF(params):
    factor,l2 = params
    model = NMFRecommender(train)
    model.fit(num_factors=factor,l1_ratio = l2)
    out = - ndcg_knn (train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_SLIM(params):
    topK,l1_ratio,alpha = params
    model = SLIMElasticNetRecommender(train)
    model.fit(topK = topK, l1_ratio = l1_ratio, alpha =alpha)
    out = - ndcg_knn (train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_EASE(params):
    factor,l2 = params
    model = EASE_R_Recommender(train)
    model.fit(l2_norm = l2)
    out = - ndcg_knn (train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_IKNN(params):
    neighbors,shrink = params #ALS
    model = ItemKNNCFRecommender(train)
    model.fit(topK=neighbors, shrink=shrink, similarity='cosine', normalize=True, feature_weighting = "TF-IDF")
    out = - ndcg_knn (train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def objective_UKNN(params):
    neighbors,shrink = params
    model = UserKNNCFRecommender(train)
    model.fit(topK=neighbors, shrink=shrink, similarity='cosine', normalize=True, feature_weighting = "TF-IDF")
    out = - ndcg_knn (train,validation,model,k=10)
    if np.abs(out + 1) < 0.001 or out < -1.0:
        return 0.0
    else:
        return out

def tune(objective,train,validation,hyperparam,n_calls,n_jobs=10):
    search_space = hyperparam
    results = forest_minimize(objective, search_space, n_calls=n_calls,random_state=1,n_jobs=n_jobs,verbose=False)

    return results

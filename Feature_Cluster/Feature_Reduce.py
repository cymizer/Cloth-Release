import numpy as np
import configure as cfg
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import time

def feature_reduce(feature, feature_type, importance=cfg.PCA_Importance, save=True):
    f"""
    Feature Dimension Reduction 

    inputs : (features,feature_type)
        e.g. (features=material,feature_type='material')

    outputs : pca_feature
    """
    #print(f"start to dimension reduce")

    f_PCA = PCA(n_components=importance)
    stime = time.time()
    f_PCA.fit(feature)
    etime = time.time()
    tot = etime - stime
    print(f"PCA fitting time : {tot:.3f}")
    
    stime = time.time()
    all_pca_feature =  f_PCA.transform(feature)
    etime = time.time()
    tot = etime - stime
    print(f"PCA transform time : {tot:.3f}")
    #print(f"finish......")
    if save:
        joblib.dump(f_PCA, f"{cfg.Model_Root}/{feature_type}_PCA.pkl")
        np.save(f'{cfg.Feature_Root}/{feature_type}_pca.npy', all_pca_feature)

    return all_pca_feature

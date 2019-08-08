import numpy as np
import configure as cfg
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib


def feature_reduce(feature, feature_type, importance=cfg.PCA_Importance, save=True):
    f"""
    Feature Dimension Reduction 

    inputs : (features,feature_type)
        e.g. (features=material,feature_type='material')

    outputs : pca_feature
    """
    #print(f"start to dimension reduce")

    f_PCA = PCA(n_components=importance)
    all_pca_feature = f_PCA.fit_transform(feature)

    #print(f"finish......")
    if save:
        joblib.dump(f_PCA, f"{cfg.Model_Root}/{feature_type}_PCA.pkl")
        np.save(f'{cfg.Feature_Root}/{feature_type}_pca.npy', all_pca_feature)

    return all_pca_feature

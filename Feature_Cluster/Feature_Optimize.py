import os
import configure as cfg
import numpy as np
from fishervector import FisherVectorGMM


def feature_optimize(features, feature_type, save=True):
    """
    feature_type : material (2048) , texture(2048) ,color(512)

    inputs : (features,feature_type)
            e.g. (features=material,feature_type='material')
    outputs : fisher vectors

    """
    features_size = features.shape[1]
    features = np.reshape(features, (-1, 1, features_size))

    print(f"All {feature_type} shape: {features.shape}\n")

    fv_gmm = FisherVectorGMM(n_kernels=cfg.GMM_Kernel).fit(
        features, model_dump_path=cfg.Model_Root+'/'+feature_type + '_GMM', verbose=False)

    print(f"{feature_type} GMM training completed!!!\n")

    all_fv = []
    for f in features:
        fv = fv_gmm.predict(np.expand_dims(f, axis=0))
        fv = fv.flatten()
        all_fv.append(fv)
    all_fv = np.asarray(all_fv)
    print(f"All {feature_type} fisher vector shape: {all_fv.shape}\n")
    if save:
        np.save(f"{cfg.Feature_Root}/{feature_type}_fv.npy", all_fv)
    return all_fv


if __name__ == "__main__":
    material = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_material.npy')
    _ = feature_optimize(material, 'material')

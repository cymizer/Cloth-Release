from keras.models import load_model, Model
from Utils import save_result
from Feature_Cluster.Feature_Extract import feature_extract
from Feature_Cluster.Feature_Optimize import feature_optimize
from Feature_Cluster.Feature_Reduce import feature_reduce
from Feature_Cluster.Feature_Cluster_Visualize import feature_cluster, feature_visualize
# from Clusters_Classification.Train import train
# from Clusters_Classification.Test import test
# from numba import cuda
from keras.backend import clear_session
import configure as cfg
import time
import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def work_flow(img_root,pattern):
    features = None
    if pattern[0]:
        # material
        material_model = load_model(cfg.Model_Root+'/material_model.h5')
        material_model = Model(inputs=material_model.input, outputs=material_model.get_layer(
        "global_average_pooling2d_1").output)
        print("Material Features Extraction...")
        start = time.time()
        material = feature_extract(img_root,material_model, 'material')
        end = time.time()
        etime = end - start
        print(f"Extract time : {etime : .3f}\n")

        start = time.time()
        material_fv = feature_optimize(material, 'material')
        end = time.time()
        otime = end-start
        print(f"Optimize time : {otime :.3f}\n")

        start = time.time()
        material_pca = feature_reduce(material_fv, 'material')
        end = time.time()
        rtime = end - start
        print(f"Reduce time : {rtime : .3f}\n")
        del material, material_fv        
        features = material_pca
        clear_session()

    if pattern[1]:
        # texture
        texture_model = load_model(cfg.Model_Root+'/texture_model.h5')
        texture_model = Model(inputs=texture_model.input, outputs=texture_model.get_layer(
        "global_average_pooling2d_1").output)

        print("Texture Features Extraction...")
        start = time.time()
        texture = feature_extract(img_root,texture_model,'texture')
        end = time.time()
        etime = end - start
        print(f"Extract time : {etime : .3f}\n")

        start = time.time()
        texture_fv = feature_optimize(texture, 'texture')
        end = time.time()
        otime = end-start
        print(f"Optimize time : {otime :.3f}\n")

        start = time.time()
        texture_pca = feature_reduce(texture_fv, 'texture')
        end = time.time()
        rtime = end - start
        print(f"Reduce time : {rtime :.3f}\n")
        
        if features is None:
            features = texture_pca
        else:    
            features = np.concatenate((features, texture_pca), axis=1)
        del texture, texture_fv
        clear_session()
        
    if pattern[2]:
        # color
        print("Color Features Extraction...")
        start = time.time()
        color = feature_extract(img_root, 'color')
        color_pca = feature_reduce(color, 'color')
        end = time.time()
        col_time = end - start
        print(f"Extract time : {col_time:.3f}")
        if features is None:
            features = color_pca
        else:    
            features = np.concatenate((features, color_pca), axis=1)

    #######################################################################

    # # concatenate features
    # features = np.concatenate((material_pca, texture_pca, color_pca), axis=1)

    # K-Means Clustering
    print("Features Clustering Computing...")
    start = time.time()
    kmeans = feature_cluster(features)
    end = time.time() 
    ctime = end - start
    print(f"Clustering : {ctime :.3f}")

    # Save Img Result
    save_result(cfg.IMG_Root, kmeans.labels_, cfg.Cluster_result)

    # Visaulize
    print("Visaulize Clustering Result...")
    start = time.time()
    feature_visualize(features, kmeans,draw=False)
    end = time.time()
    vtime = end-start
    print(f"Visualize : {vtime :.3f}")

    # # Classification
    # start = time.time()
    # model = train(features, kmeans.labels_, epoch=25, draw=True)
    # end = time.time()
    # cls_train = end - start
    # print(f"Classification train time : {cls_train:.3f}")

    # start = time.time()
    # test(features, kmeans.labels_, model)
    # end = time.time()
    # cls_test = end - start
    # print(f"Classifiction test time : {cls_test:.3f}")


if __name__ == "__main__":
    work_flow(cfg.IMG_Root,cfg.Pattern)
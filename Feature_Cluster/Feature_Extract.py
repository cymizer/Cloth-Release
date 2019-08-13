from numba import cuda
from keras.models import load_model, Model
import numpy as np
import os
import cv2
import glob
import configure as cfg
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# assign GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占滿GPU Memory, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

# disable Debug Information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def feature_extract(img_root, feature_type='color', save=True):
    """
    feature_type : material (2048) , texture(2048) ,color(512)

    inputs : (img_root_dir,feature_type)
    outputs : features

    """
    if feature_type == 'material':
        State = 0
        model = load_model(cfg.Model_Root+'/material_model.h5')
        model = Model(inputs=model.input, outputs=model.get_layer(
            "global_average_pooling2d_1").output)
    elif feature_type == 'texture':
        State = 0
        model = load_model(cfg.Model_Root+'/texture_model.h5')
        model = Model(inputs=model.input, outputs=model.get_layer(
            "global_average_pooling2d_1").output)
    elif feature_type == 'color':
        State = 1

    all_img_file = glob.glob(img_root+'/*/*.jpg')
    all_feature = []
    if not State:
        for index, i in enumerate(all_img_file):
            img = cv2.imread(i)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)/255
            f = model.predict(img)
            all_feature.append(f)
            if index % cfg.LOG_INTERVAL == 0 and (index != 0):
                print(
                    f'{index} / {len(all_img_file)} ({(index/len(all_img_file))*100 : .2f} %)')
        all_feature = np.asarray(all_feature, dtype=np.float32)
        all_feature = np.reshape(all_feature, (-1, 2048))
    else:
        for i in all_img_file:
            img = cv2.imread(i)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [
                                0, 256, 0, 256, 0, 256])  # 512 dim
            #print ("3D histogram shape: %s , with %d values" % (hist.shape, hist.flatten().shape[0]))
            #print ("image shape :", img.shape)
            hist = hist/(img.shape[0]*img.shape[1])
            all_feature.append(hist.flatten())
            if index % cfg.LOG_INTERVAL == 0 and (index != 0):
                print(
                    f'{index} / {len(all_img_file)} ({(index/len(all_img_file))*100 : .2f} %)')
        all_feature = np.asarray(all_feature, dtype=np.float32)
        all_feature = np.reshape(all_feature, (-1, 512))
    if save:
        np.save(f"{cfg.Feature_Root}/{feature_type}.npy", all_feature)
    KTF.clear_session()
    if feature_type != 'color':
        del model

    return all_feature


if __name__ == "__main__":

    f = feature_extract(img_root=cfg.IMG_Root, feature_type='material')
    # f_model = load_model('Z:/Users/cymb103u/Desktop/WorkSpace/Cloth-Release/Model/material_model.h5')
    # all_img_file = glob.glob('Z:/Users/cymb103u/Desktop/Dataset/Img/female/*/*.jpg')
    # all_feature = []
    # for i in all_img_file:
    #     img = cv2.imread(i)
    #     img = cv2.resize(img, (224, 224))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = np.expand_dims(img, axis=0)/255
    #     f = f_model.predict(img)
    #     all_feature.append(f)
    # all_feature = np.asarray(all_feature, dtype=np.float32)
    # all_feature = np.reshape(all_feature, (-1, 2048))

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


def Visaulize(model, img):
    # layers = [171]
    r_model = Model(inputs=model.inputs, outputs=model.layers[171].output)
    feature = r_model.predict(img)
    print(feature.shape)


if __name__ == "__main__":

    f_model = load_model(cfg.Model_Root+'/material_model.h5')
    img = cv2.imread(
        'Z:/Users/cymb103u/Desktop/Dataset/HIPSTER/Goth/17382.jpg')
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)/255

    Visaulize(f_model, img)
    # f_model.summary()
    pass

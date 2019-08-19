import configure as cfg
import glob
import joblib
import os
from shutil import copyfile
# import pathlib


def save_result(img_root, label_result, save_root):
    img_path = glob.glob(img_root+'/*/*.jpg')
    # save
    NUM = 0
    while True:
        try:
            os.makedirs(save_root+f'/new_{NUM:03d}')
            save_path = save_root+f'/new_{NUM:03d}'
            break
        except FileExistsError:
            NUM += 1
    print(save_path)

    # make save_result folder
    cluster_num = max(label_result)
    for i in range(cluster_num+1):
        os.makedirs(save_path+f'/{i:02d}')

    # copy
    for img, lab in zip(img_path, label_result):
        file_name = img.split('\\')[2]
        copyfile(src=img, dst=save_path+f'/{lab:02d}/{file_name}')


if __name__ == "__main__":
    kms = joblib.load('Z:/Users/cymb103u/Desktop/WorkSpace/Cloth-Release/Model/K-means.pkl')
    save_result(cfg.IMG_Root, kms.labels_, cfg.Cluster_result)

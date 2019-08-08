# 衣著流行元素分析說明文件

## 環境需要
    python==3.6
    tensorflow-gpu==1.8.0
    keras==2.2.4
    sklearn
    fishervector
    matplotlib 
    numba
    opencv-python
    seaborn
    pytorch 1.1


## 環境安裝步驟
1. 安裝 [python3.6](https://www.python.org/downloads/) 
2. 使用 cmd 移動到 Cloth-Release 資料夾下
3. 在 cmd 中輸入 `pip install -r requirements.txt `
4. 安裝 [pytorch 1.1](https://pytorch.org/get-started/locally/) 
    
    `pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-win_amd64.whl`
    
----------------------------------------------------------------------

## 執行步驟
### 參數修改
- 在 Cloth-Release/configure.py 修改圖片根目錄位置(IMG_Root)
### cmd 執行
1. 使用 cmd 移動到 Cloth-Release 資料夾下
2. 在 cmd 輸入 `python main.py`
3. 得到執行結果

## 程式碼文件說明

### Cloth-Release/
* main.py :主要運行程式，workflow()包含整個架構運行流程。
     
* configure.py : 參數設定、儲存位置，可在此處修改cluster數量


### Cloth-Release/Clusters_Classification
* feature_cls_model.py : feature 分類模型

* feature_dataset.py : 資料載入器

    inputs : features , feature_labels 

* myconfig.py : 深度學習模型參數

* Train.py : 訓練模型

    inputs : features, feature_labels
    
    outputs : model

    註: 這邊feature_labels 是利用 kmeans過後結果所得
* Test.py : 測試訓練完之模型

    inputs : features, feature_labels, model

### Cloth-Release/Feature_Clsuter
* Feature_Extract.py : 對image取feature,可以設定不同的feature_type(e.g. material, texture)


    inputs : img_root, feature_type
    ```
    feature_type : material, texture, color
    ```
    outputs : extract_feature
* Feature_Optimize.py : 將取完的feature做特徵優化

    inputs : extract_feature, feature_type
    
    outputs : fisher_vector

* Feature_Reduce.py : 將優化特徵進行PCA降維

    inputs : fisher_vector, feature_type 
    outputs : pca_feature

* Feature_Cluster_Visualize.py : 將特徵進行clustering,得到clustering label。並且將結果 visualize。

    1. feature_cluster()

        inputs : features, clusters
        ```
        clusters: 聚類數量可自由調整，在 Cloth-Release/configure.py
        ```
        output : kmeans
        
    
    2. feature_visualize():
        inputs : features, kmeans


## Project file Structure
    Cloth-Release
    │
    ├── configure.py
    ├── main.py
    ├── README.md
    ├── requirements.txt
	│
    ├── Clusters_Classification
    │   ├── feature_cls_model.py
    │   ├── feature_dataset.py
    │   ├── Train.py
    │   ├── Test.py
    │   └── myconfig.py
    │
    ├── Feature
    │   └──....
    ├── Model
    │   └──....
    ├── Result
    │   └──....
    │
    └── Feature_Clusters
        ├── Feature_Cluster_Visualize.py
        ├── Feature_Extract.py
        ├── Feature_Optimize.py
        └── Feature_Reduce.py


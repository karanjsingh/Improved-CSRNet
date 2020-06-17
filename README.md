## IIT Tirupati
# CSRNet keras implementation for crowd count

Team : Dr. Rama Krishna Gorthi, Karanjeet singh (ML/DL)

# Weights/Model/TestFile/Thesis/ppt: https://drive.google.com/drive/folders/1YcM01mrJR3aHKDDNdMvwaHFHpfg2h2zi?usp=sharing
# CSRNet Paper(CVPR 2018) : https://arxiv.org/abs/1802.10062
# Pytorch Implementation : https://github.com/leeyeehoo/CSRNet-pytorch

# Dataset :
The dataset ShanghaiTech: https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view

The dataset have Part A which consists of images with a high density of crowd or dense crowd scenes and Part B which consists of images with sparse crowd scenes.   

Both parts of the dataset were trained and we have 2 different models with the same architecture, both for dense and sparse regions for 200 epochs. Other hyperparameters are kept same as specified in the CSRNet paper and pytorch implementation.

## RUN TEST FILE USING COMMAND---> python test_count.py --image path_of_testImage --weight path_of_weight
## e.g --> python test_count.py --image test_images/iittp_convocation_268.jpg --weight weights/sparse.h5


## Requirements :

1. Numpy : 1.14.3
2. Scipy : 1.1.0
3. Tensorflow : 1.9.0
4. Keras : 2.2.2
5. Pillow(PIL) : 5.1.0
6. OpenCV : 3.4.1



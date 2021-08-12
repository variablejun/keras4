import numpy as np
from sklearn.decomposition import PCA

from tensorflow.keras.datasets import mnist

(x_train, _),(x_test, _) = mnist.load_data() #_ 무시하겟다 받지않겟다.
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test,axis=0)
print(x.shape) # (70000, 28, 28)
pca = PCA(n_components=20) # 랜덤이 아닌 자기 기준에 따라서 압축을 시킨다 -> 제거하는것은 아님
x = x.reshape(70000,28*28)
x = pca.fit_transform(x)
print(x)
# 0.95 이상의 컴포넌트
'''
[[ 122.25517165 -316.23368444  -51.13226141 ...  189.13518328
  -359.14855144  203.35451841]
 [1010.49413581 -289.96378239  576.11995886 ... -102.43304396
    35.30899721  152.46015974]
 [ -58.99588479  393.69737788 -161.99794997 ...  374.15900464
   -78.34089854 -192.87282709]
 [-271.50712743  590.07864385  341.36965886 ... -344.73143844
   -52.580183   -804.9970655 ]
 [-310.22497929 -116.72684573  635.72039899 ...  145.08508431
   284.64359653 -205.87863673]
 [1058.862156    -83.39272251  731.34142955 ... -421.52356969
   141.45880077   67.78709357]]
'''

pcaEVR = pca.explained_variance_ratio_
print(pcaEVR) # 
cunsum = np.cumsum(pcaEVR)
print(cunsum)
print(np.argmax(cunsum >= 0.95)+1) # 7




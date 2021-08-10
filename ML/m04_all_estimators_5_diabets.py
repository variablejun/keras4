import numpy as np 
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore') # 워닝 무시

allAllalgorythsm1 = all_estimators(type_filter='classifier') # 분류 모델 모델의 개수는  41
allAllalgorythsm2 = all_estimators(type_filter='regressor') # 회귀모델 모델의 개수는  54

print(allAllalgorythsm1)
print(allAllalgorythsm2)
print('모델의 개수는 ',len(allAllalgorythsm2))  
for (name, algorythsm) in allAllalgorythsm1:
     try :
          model = algorythsm()
          model.fit(x_train,y_train)
          y_predict = model.predict(x_test)
          acc = accuracy_score(y_test,y_predict)
          print(name,'의 acc : ',acc)
     except:
          print(name,'없는것')


'''
분류모델로 돌릴경우 값도 제대로 안나오고 속도도 많이 느리다.
MinMaxScaler

===============================================================

StandardScaler

'''
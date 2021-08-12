# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,dataset.target,
train_size = 0.7, random_state=76)

#model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test,y_test)
print('acc :', acc)
print(model.feature_importances_) # iris의 컬럼중에서 어느 컬럼이 영향이큰지 영향도를 나타내는것
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
     n_feature = dataset.data.shape[1]
     plt.barh(np.arange(n_feature),model.feature_importances_,align='center')
     plt.yticks(np.arange(n_feature),dataset.feature_names)
     plt.xlabel('feature_importances')
     plt.ylabel('features')
     plt.ylim(-1,n_feature)

plot_feature_importances_dataset(model)   
plt.show()
'''
DecisionTreeClassifier
acc : 0.8888888888888888
[0.01433251 0.03439803 0.909934   0.04133545]

max_depth=4
acc : 0.8888888888888888
[0.         0.03520536 0.4505695  0.51422514]


RandomForestClassifier
acc : 0.9777777777777777
[0.07979185 0.03721278 0.49060171 0.39239365]
'''

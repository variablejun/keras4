import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential 
#from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D

dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape) # (569, 30) (569,) y 0 과 1로 구성
# 원핫 인코딩
'''

'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(142,4,1,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(8, 4, 1,1)
'''
(150, 4) (150,)
(8, 4) (142, 4)
'''
model = Sequential() 
model.add(Conv2D(256,3 ,padding = 'same' ,input_shape=(4,1,1)))
model.add(Conv2D(128,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(32,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(16,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(1,3,padding = 'same' ,activation = 'relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3, activation='softmax')) # 이진분류모델
'''

'''
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=20, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.25, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
콘보루션1D
걸린시간 3.3642373085021973
loss :  0.7070646286010742
accuracy :  0.375

걸린시간 3.3630592823028564
loss :  1.0366997718811035
accuracy :  0.5

콘보루션 2D
걸린시간 3.3504817485809326
loss :  0.3988201320171356
accuracy :  1.0
'''
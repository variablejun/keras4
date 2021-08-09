import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
# acc 0.8이상

x = np.load('./_save/_npy/k55_x_data_wine.npy')
y = np.load('./_save/_npy/k55_y_data_wine.npy')

print(x.shape , y.shape) # (178, 13) (178,)
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3, activation='softmax'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)
reduce_LR = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',verbose=1,factor=0.5) 

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy']) # 이진분류모델 에 대한 로스
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.3, callbacks=[es,reduce_LR]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])


print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

'''

loss :  1.854365621056786e-07
accuracy :  1.0

loss :  1.3245475827261544e-08
accuracy :  1.0


Reduce
0.1
loss :  0.13655872642993927
accuracy :  0.8888888955116272
0.01
loss :  0.18203169107437134
accuracy :  0.8888888955116272
0.001
loss :  0.013466281816363335
accuracy :  1.0
'''
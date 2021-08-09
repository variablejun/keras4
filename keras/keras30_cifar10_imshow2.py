#모델링완성 하시오 32,32,3
# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(50000, 32, 32, 3)/255 # 255는 픽셀의 범위, 전처리
x_test = x_test.reshape(10000, 32, 32, 3)/255

'''
  ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 1 but received input with shape (None, 32, 32, 3)

(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
'''

ValueError: Expected 2D array, got 1D array instead:
array=[7 2 1 ... 4 5 6].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample
'''
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(10000, 10)
(50000, 10)
'''
# Conv2D에 들어가 연산하기 위해 3차원 데이터를 4차원 데이터로 쉐입을 바꾸어준다. 



# 0.992 이상으로만들기 , 캡쳐후 단톡에 전송
model = Sequential() 
model.add(Conv2D(64, kernel_size=(2,2) ,padding = 'same' ,input_shape=(32,32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(16, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(8, (2,2),padding = 'same' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())#(N, 180)
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='sigmoid')) # 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=20, mode='max', verbose=3)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''


model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
Epoch 00142: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 4.4896 - accuracy: 0.6071
loss :  4.489560127258301
accuracy :  0.6071000099182129

model = Sequential() 
model.add(Conv2D(64, kernel_size=(2,2) ,padding = 'same' ,input_shape=(32,32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(16, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(8, (2,2),padding = 'same' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(8, (2,2),padding = 'same' ,activation = 'relu'))
model.add(MaxPooling2D())
loss :  2.5426201820373535
accuracy :  0.5964999794960022

정규화 추가후
loss :  1.3356002569198608
accuracy :  0.5568000078201294

loss :  2.6508970260620117
accuracy :  0.5257999897003174

loss :  2.264362335205078
accuracy :  0.522599995136261

loss :  2.466057538986206
accuracy :  0.5016000270843506

loss :  2.6687827110290527
accuracy :  0.46650001406669617

'''
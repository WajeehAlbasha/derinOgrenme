# -*- coding: utf-8 -*-

from keras.datasets import cifar100
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train1 = []
x_train1 = []
y_test1 = []
x_test1 = []

for i in range (50000):
    if y_train[i] == 7 : 
        y_train1.append(0)
        x_train1.append(x_train[i])
        
    elif  y_train[i] == 16 : 
        y_train1.append(1)
        x_train1.append(x_train[i])
        
    elif  y_train[i] == 37 : 
        y_train1.append(2)
        x_train1.append(x_train[i])
        
    elif  y_train[i] == 50 : 
        y_train1.append(3)
        x_train1.append(x_train[i])
        
    elif  y_train[i] == 58 : 
        y_train1.append(4)
        x_train1.append(x_train[i])
        
    elif  y_train[i] == 73 : 
        y_train1.append(5)
        x_train1.append(x_train[i])
            
    elif  y_train[i] == 83 : 
        y_train1.append(6)
        x_train1.append(x_train[i])
       
y_train1= np.array(y_train1)
x_train1= np.array(x_train1)

for i in range (10000):
    if y_test[i] == 7 : 
        y_test1.append(0)
        x_test1.append(x_test[i])
        
    elif  y_test[i] == 16 : 
        y_test1.append(1)
        x_test1.append(x_test[i])
        
    elif  y_test[i] == 37 : 
        y_test1.append(2)
        x_test1.append(x_test[i])
        
    elif  y_test[i] == 50 : 
        y_test1.append(3)
        x_test1.append(x_test[i])
        
    elif  y_test[i] == 58 : 
        y_test1.append(4)
        x_test1.append(x_test[i])
        
    elif  y_test[i] == 73 : 
        y_test1.append(5)
        x_test1.append(x_test[i])
            
    elif  y_test[i] == 83 : 
        y_test1.append(6)
        x_test1.append(x_test[i])
        
y_test1= np.array(y_test1)
x_test1= np.array(x_test1)

index=[]
enum=0
for i in range (2500):
    if y_train1[i] == 0 and enum < 10: 
        index.append(i)
        enum=enum+1
        if enum == 10 : i = 0
    elif y_train1[i] == 1 and enum >= 10 and enum < 20:
        index.append(i)
        enum=enum+1
        if enum == 20 : i = 0
    elif y_train1[i] == 2 and enum >= 20 and enum < 30:
        index.append(i)
        enum=enum+1
        if enum == 30 : i = 0           
    elif y_train1[i] == 3 and enum >= 30 and enum < 40: 
        index.append(i)
        enum=enum+1 
        if enum == 40 : i = 0           
    elif y_train1[i] == 4 and enum >= 40 and enum < 50: 
        index.append(i)          
        enum=enum+1
        
        if enum == 50 : i = 0           
    elif y_train1[i] == 5 and enum >= 50 and enum < 60: 
        index.append(i)          
        enum=enum+1
        
        if enum == 60 : i = 0           
    elif y_train1[i] == 6 and enum >= 60 and enum < 70: 
        index.append(i)          
        enum=enum+1
        
for i in range(0,70):
    plt.subplot(7,10,i+1)
    plt.imshow(x_train1[index[i]])
    fig = plt.imshow(x_train1[index[i]])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
x_train1=x_train1.astype('float32')/255.0
x_test1=x_test1.astype('float32')/255

y_train1=to_categorical(y_train1,7)
y_test1=to_categorical(y_test1,7)

model=Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape= (32, 32, 3)))
model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(7,activation='softmax'))
model.summary()

from tensorflow.keras import optimizers
from keras import losses
model.compile(loss=losses.CategoricalCrossentropy(),
              optimizer=optimizers.Adam(lr=1e-6),
              metrics=['acc'])

history=model.fit(x_train1,
                  y_train1,
                  epochs=80,
                  validation_data=(x_test1,y_test1))

test_loss, test_acc = model.evaluate(x_test1,y_test1)
print(" Loss = %",100*test_loss,"\n","Accuracy = %",100*test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

pred=model.predict(x_test1)


for i in range(0,700):
    maxindex= np.argmax(pred[i], axis=None, out=None)
    for y in range(0,7):
        if y == maxindex:
            pred[i,y] = 1
        else :
            pred[i,y] = 0

from sklearn.metrics import confusion_matrix, classification_report


print('Confusion Matrix')
print(confusion_matrix(y_test1.argmax(axis=1), pred.argmax(axis=1)))
print("\n",'Classification Report')
target_names = ["beetle", "can", "house", "mouse", "pickup_truck", "shark", "sweet_pepper"]
print(classification_report(y_test1, pred, target_names=target_names))

# test kodu :
#    test_list = [x_train1[index[8]],x_train1[index[12]],x_train1[index[26]],x_train1[index[31]],x_train1[index[48]],x_train1[index[53]],x_train1[index[69]]]

#    test_pred = model.predict(np.array(test_list))

#    plt.imshow(test_list[0])
#    fig = plt.imshow(test_list[0])
#    fig.axes.get_xaxis().set_visible(False)
#    fig.axes.get_yaxis().set_visible(False)
#    for j in range (7) :
#    print("Sınıf", j+1, "=" , target_names[j], "\t\t%", 100* test_pred[0,j] )

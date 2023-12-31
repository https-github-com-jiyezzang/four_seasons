#!/usr/bin/env python
# coding: utf-8

# # ResNet50

# # 웜:쿨 비율 동일x
# warm_1 740장 + warm_2 2307장 + cool_1 238장 + cool_2 336장, 3007:574, 총 3621장



# ## (1) 라이브러리 및 데이터 불러오기

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import vgg16

import numpy as np
import cv2
import os
import glob
from PIL import Image



# ## (2) 데이터 불러오고, 웜톤 0, 쿨톤 1로 레이블링

def load_data(img_path, number_of_data=3621):  # warm_1 740 + warm_2 2307 + cool_1 238 + cool_2 336
    # 웜톤 : 0, 쿨톤 : 1
    img_size=224
    color=3
    #이미지 데이터와 라벨(웜톤 : 0, 쿨톤 : 1) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0    
    
    warm_files = (list(glob.iglob(img_path + '/warm_1/*.jpg')) + list(glob.iglob(img_path + '/warm_1/*.JPG')) +
              list(glob.iglob(img_path + '/warm_2/*.jpg')) + list(glob.iglob(img_path + '/warm_2/*.JPG')))

    for file in warm_files:
        img = Image.open(file)  # 이미지 열기
        img = img.resize((img_size, img_size))  # 이미지 크기 조정
        img = np.array(img, dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 웜톤 : 0
        idx=idx+1

                
    cool_files = (list(glob.iglob(img_path + '/cool_1/*.jpg')) + list(glob.iglob(img_path + '/cool_1/*.JPG')) +
              list(glob.iglob(img_path + '/cool_2/*.jpg')) + list(glob.iglob(img_path + '/cool_2/*.JPG')))

    
    for file in cool_files:
        img = Image.open(file)  # 이미지 열기
        img = img.resize((img_size, img_size))  # 이미지 크기 조정
        img = np.array(img, dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 쿨톤 : 1
        idx=idx+1  
    
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/project/first-repository/aiffelthon/content/drive/MyDrive/사계절_연예인 이미지 데이터셋/train data"
(x_train, y_train)=load_data(image_dir_path)

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))



# ## (3) 정규화

x_train = x_train / 255.0



# ## (4) train, val 분리하기

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.3,
                                                  random_state=42)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)



# ## (5) 사전학습 모델 불러오기

# 사전학습 ResNet50 불러오기
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras import optimizers, initializers, regularizers, metrics


# ResNet-50 모델 불러오기
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = GlobalAveragePooling2D()(resnet50.output) 
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output = Dense(2, activation='softmax')(x)


# 새로운 모델 정의
model = tf.keras.models.Model(inputs=resnet50.input, outputs=output)



# ## (6) 모델 컴파일

learning_rate = 0.0001
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


model.compile(optimizer= adam,
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])



# ## (7) early stopping, checkpoint
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True,
)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    './check/ResNet/4번/2-1-2/model_{epoch:02d}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch',    
)



# ## (8) 모델 학습하기

# early stopping과 checkpoint가 적용된 model.fit
history = model.fit(
    x_train, y_train, 
    batch_size=32, 
    epochs=50, 
    validation_data=(x_val, y_val), 
    callbacks=[model_checkpoint_callback, early_stopping_callback]
)



# ## (9) 예측하기

pred = model.predict(x_val)
pred_class = np.argmax(pred, axis=1)

print(pred_class)
print(y_val)
print(pred)

acc = np.mean(pred_class == y_val)
print('accuracy: %f' % (acc,))



# ## (10) val_loss, val_accuracy 시각화

import matplotlib.pyplot as plt

# 정확도 시각화
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 손실값 시각화
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



# ## (11) confusion_matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# confusion matrix 생성
cm = confusion_matrix(y_val, pred_class)

# confusion matrix 출력
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()



# ## (12) classification_report

from sklearn.metrics import classification_report

print(classification_report(y_val, pred_class))

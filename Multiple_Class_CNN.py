import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Activation, Conv2D, MaxPooling2D


#from PIL import Image, ImageDraw
from imutils import paths
from tensorflow.keras.utils import to_categorical

import Data_PreProcess as DPP

id_line = []

def trainData():
    trainData_Dir = 'Birds\\train'
    trainData_Path = os.path.join(trainData_Dir)
    train_DPP = DPP.Data_Category(trainData_Path)
    categories = train_DPP.create_category()
    cat_count = len(categories)
    print('Train Categories counted: ',cat_count)
    trainX,trainY = train_DPP.create_Data()
    trainX = np.array(trainX).reshape(-1,96,96,3)
    trainY = np.array(trainY)

    #! normalize data
    trainX = trainX/255.0

    return trainX,trainY,categories

def testData():
    testData_Dir = 'Birds\\test'
    testData_Path = os.path.join(testData_Dir)
    test_DPP = DPP.Data_Category(testData_Path)
    categories = test_DPP.create_category()
    cat_count = len(categories)
    print('Test Categories counted: ',cat_count)
    testX,testY = test_DPP.create_Data()
    testX = np.array(testX).reshape(-1,96,96,3)
    testY = np.array(testY)

    #! normalize data
    testX = testX/255.0

    return testX,testY,categories

def model_train(x,y,categories,testX,testY):
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=x.shape[1:])) 
    #? 過濾器數量filters, 指定卷積大小的高與寬kernel_size, 步長strides(default = 1), padding卷積如何處理邊緣(選項包含 valid 與 same，default = valid), 激活函數activation, 指定輸入層高度input_shape
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #? 最大池化的窗口大小pool_size(為整數，沿垂直、水平方向縮小比例的因數，若只有一個整數時代表兩個維度都會使用同樣的比例)
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x,y,epochs=75,batch_size=32,validation_data=(testX, testY))

    model.save('saved_model/MultiClass_CNN')

def UseSavedModel():
    testData_Dir = 'Birds\\test'
    testData_Path = os.path.join(testData_Dir)
    test_DPP = DPP.Data_Category(testData_Path)
    categories = test_DPP.create_category()
    model = tf.keras.models.load_model('saved_model/MultiClass_CNN')

    x_test = []
    id_line = []
    imagePaths = sorted(list(paths.list_images(testData_Path) ) )
    for imgpath in imagePaths:    

        img_array = cv2.imread(imgpath,cv2.IMREAD_COLOR)
        new_img_array = cv2.resize(img_array,dsize=(96,96))

        label = imgpath.split(os.path.sep)[-2]
        id_line.append(label)

        x_test.append(new_img_array)
    x_test = np.array(x_test).reshape(-1,96,96,3)
    #! normalize
    x_test = x_test/255.0

    predictions = model.predict(x_test)


    print('len of ID:',len(id_line),'id value:',id_line[0])
    print('len of predictions:',len(predictions),'predictions[0]:',predictions[0])
    #print(np.argsort(predictions[-2])[:2])
    prob = [p[np.argmax(p)]*100 for p in predictions]
    print('len of prob:',len(prob),'prob[0]:',prob[0])
    predict_value = [categories[np.argmax(p)] for p in predictions]
    print('len of predic_value:',len(predict_value),'predict_value[0]:',predict_value[0])
    df = pd.DataFrame({'id':id_line,'predict value':predict_value,'Probability':prob})
    df.to_csv('hw1_prediction.csv',index = False)

def predict_grading():
    testData_Dir = 'Birds\\test'
    testData_Path = os.path.join(testData_Dir)
    test_DPP = DPP.Data_Category(testData_Path)
    categories = test_DPP.create_category()

    model = tf.keras.models.load_model('saved_model/MultiClass_CNN')

    x_grade = []
    id_line = []
    gradeData_Dir = 'Birds\\grading_data'
    imagePaths = list( paths.list_images(gradeData_Dir) )
    for imgpath in imagePaths:    
        img_array = cv2.imread(imgpath,cv2.IMREAD_COLOR)
        new_img_array = cv2.resize(img_array,dsize=(96,96))

        label = imgpath.split(os.path.sep)[-1]
        id_line.append(label)
        x_grade.append(new_img_array)

    x_grade = np.array(x_grade).reshape(-1,96,96,3)
    #! normalize
    x_grade = x_grade/255.0

    predictions = model.predict(x_grade)


    print('len of ID:',len(id_line),'id value:',id_line[0])
    print('len of predictions:',len(predictions),'predictions[0]:',predictions[0])
    #print(np.argsort(predictions[-2])[:2])
    prob = [p[np.argmax(p)]*100 for p in predictions]
    print('len of prob:',len(prob),'prob[0]:',prob[0])
    predict_value = [categories[np.argmax(p)] for p in predictions]
    print('len of predic_value:',len(predict_value),'predict_value[0]:',predict_value[0])
    '''
    Excel split number string from file name, then convert string to value:
    #? =VALUE(LEFT(A1, SEARCH(".",A1,1)-1))
    '''
    df = pd.DataFrame({'image':id_line,'species':predict_value})
    df.to_csv('grading_pred.csv',index = False)

if __name__ == '__main__':

    '''#! follow code(line 135 - line 137) is to fix the error of tensorflow gpu "tensorflow.python.framework.errors_impl.NotFoundError:  No algorithm worked!" '''
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    trainX,trainY,trainCategoryList = trainData()
    testX,testY,testCategoryList = testData()
    print('TrainX:',trainX.shape,'TrainY:',trainY.shape)
    print('TestX:',testX.shape,'TestY:',testY.shape)
    model_train(trainX,trainY,trainCategoryList,testX,testY)

    #UseSavedModel()
    #predict_grading()
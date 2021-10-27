import os
import cv2 #pip install opencv-python
from sklearn.preprocessing import MultiLabelBinarizer
from imutils import paths
import random
import numpy as np

trainData_Dir = 'Birds\\train'
trainData_Path = os.path.join(trainData_Dir)


class Data_Category:
    def __init__(self,path):
        self.x = []
        self.y = []
        self.path = path
        self.categories = []
        self.mlb = MultiLabelBinarizer()

    def create_category(self):
        #? read all possible category and append them to list
        for subDir in os.listdir(self.path):
            self.categories.append((subDir))
        self.mlb.fit([self.categories])
        return self.categories
    
    def mlb_transform(self,str_in):
        #? convert categories to one-hot binary
        transform = list(self.mlb.transform([[str_in]])[0])
        return transform
        #return transform.index(1)
    
    def create_Data(self):
        print("[INFO] loading images...")
        imagePaths = sorted(list(paths.list_images(self.path) ) )
        random.seed(20)
        random.shuffle(imagePaths)
        for imgpath in imagePaths:    

            img_array = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            new_img_array = cv2.resize(img_array,dsize=(96,96))

            label = imgpath.split(os.path.sep)[-2]

            self.x.append(new_img_array)
            self.y.append(self.mlb_transform(label))
        return  self.x, self.y

'''
if __name__ == '__main__':
    #trainData_Dir = 'Birds\\train'
    #trainData_Path = os.path.join(trainData_Dir)
    tmp = Data_Category(trainData_Path)
    print(tmp.create_category())
    #print(tmp.mlb_transform('AFRICAN CROWNED CRANE'))
    #print(tmp.create_Data())
'''

'''
    [A,B,C]
A = [1,0,0]
B = [0,1,0]
C = [0,0,1]
300 = [0,0,0,.....,1]
A = [ 0.98 , 0.02 , 0]
'''
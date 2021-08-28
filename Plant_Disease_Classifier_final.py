'''
MADE BY AAYUSH DESHMUKH
ON DATE 11-03-2021
TO CLASSIFY HEALTH STATUS OF A PLANT BY LOOKING AT ITS LEAVES
Plant_Disease_Classifier.py
'''

import cv2 as cv
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



############################################################################################################################

file  = open('labels.txt', 'r')
class_names = file.read().split('\n')
print("Classifier classes are:")
for i in class_names:
    print(i)

model=load_model("Saved_models/9869_9810_Adam_Sparse_L2_Dropout")
model.summary()


test_video=input("Enter the name of the video file(with extension) stored in 'test_videos' folder which you want to classify:") 
cap= cv.VideoCapture(r'test_videos/'+test_video)
i=0
temp=[]
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    img = cv.resize(frame, (256, 256), 
               interpolation = cv.INTER_NEAREST)
    img2 = img_to_array(img)
    img2=img2.reshape([1,256,256,3])
    a=model.predict(img2)
    for j in range(len(a[0])):
        if a[0][j]==1:
            temp.append(class_names[j])
    i+=1
    
cap.release()
cv.destroyAllWindows()


temp_set = set(temp)
predictions = list(temp_set)
counts=[]
for i in predictions:
    counts.append(temp.count(i))
index = counts.index(max(counts))
print("Plant's health classified as:",predictions[index])
 







 







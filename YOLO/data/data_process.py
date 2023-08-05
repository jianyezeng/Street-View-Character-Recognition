import os
import cv2
import json

def process(dict1,shape):
    l=''
    for i in range(len(dict1['left'])):
        l += str(dict1['label'][i])+' '+\
        str((dict1['left'][i]+dict1['width'][i]/2)/shape[1])+' '+\
        str((dict1['top'][i]+dict1['height'][i]/2)/shape[0])+' '+\
        str(dict1['width'][i]/shape[1])+' '+\
        str(dict1['height'][i]/shape[0])\
              +'\n'
    return l
    
f = open(
    "val.json",
    encoding='utf-8')
data = json.load(f)
for i in data:
    img = cv2.imread(r'val/images'+'/'+i)
    shape = img.shape
    f = open(r'val/labels/'+i[0:6]+'.txt','w')
    f.write(process(data[i],shape))
    f.close()

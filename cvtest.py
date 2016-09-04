#!/Users/chenyu/anaconda/envs/py27/bin/python
import glob
from panorama import Sticher
import cv2
import sys
import re
import numpy as np

f = []
directory = "/Users/chenyu/9517test/"
f.extend(glob.glob("/Users/chenyu/9517test/*"))
max = 0
k = 0

# result = np.ones(1000, 1000)

while len(f) > 1:
    name = ("", "")
    max = 0
    for i in f:
        for j in f:
            if(i != j):
                image1 = cv2.imread(i)
                image2 = cv2.imread(j)
                sticher = Sticher()
                rank = sticher.rank(image1, image2)
                print "file1:", i, "file2:", j, rank
                if (rank > max):
                    max = rank
                    name = (i, j)
    image1 = cv2.imread(name[0])
    image2 = cv2.imread(name[1])
    sticher = Sticher()
    image3 = sticher.stich(image1, image2)
    tmpname = directory + "tmp" + str(k) + ".jpg"
    cv2.imwrite(tmpname, image3)
    f.remove(name[0])
    f.remove(name[1])
    f.append(tmpname)
    k = k + 1



#delete black surranding
# gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# x, y, w, h = cv2.boundingRect(cnt)
# crop = image3[y:y+h,x:x+w]
# cv2.imwrite("/Users/chenyu/9517test/final.jpg", crop)



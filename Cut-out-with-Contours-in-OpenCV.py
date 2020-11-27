import cv2 as cv
import numpy as np

img = cv.imread('test.jpg')
img = 255-img
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img_gray,(5,5),0)
ret,th = cv.threshold(blur,20,255,cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

pic_area = img.shape[0]*img.shape[1]
cnt_area_list = [cv.contourArea(cnt)/pic_area for cnt in contours]
area_th = 0.1
cnt_idx = np.where([c > area_th for c in cnt_area_list])

cnt = contours[0]

fP = cv.fillPoly(img.copy(),[cnt],(0,255,0))
cv.imshow('',fP)
cv.waitKey(0)
cv.destroyAllWindows()
top_left = (cnt[:,:,0].min(),cnt[:,:,1].min())
bottom_right = (cnt[:,:,0].max(),cnt[:,:,1].max())
print(top_left)
rect = cv.rectangle(img.copy(),top_left,bottom_right,(255,0,0),2,)
cv.imshow('',rect)
cv.waitKey(0)
import cv2 as cv
import numpy as np
import argparse

original = cv.imread('test.jpg')
img = 255-original
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img_gray,(5,5),0)
ret,th = cv.threshold(blur,20,255,cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
pic_area = img.shape[0]*img.shape[1]
cnt_area_list = [cv.contourArea(cnt)/pic_area for cnt in contours]
area_th = 0.1
cnt_idx = np.where([c > area_th for c in cnt_area_list])
cnt = contours[cnt_idx[0][-1]]
black_board = np.zeros(original.shape[:2])
fP = cv.fillPoly(black_board,[cnt],(255,255,255))
cut_out = cv.bitwise_and(original,original,mask=np.uint8(fP))
cut_out = cut_out[cnt[:,:,1].min():cnt[:,:,1].max(),
          cnt[:,:,0].min():cnt[:,:,0].max(),:]

cv.imshow('',cut_out)
cv.waitKey(0)
import cv2 as cv
import numpy as np
import argparse
# 用于指定参数
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',type=str,help = 'file path',
                default='test2.jpg')
# 指定轮廓面积占图片比例的阈值，超过阈值的轮廓才能被显示
ap.add_argument('-th','--threshold',type = float,help = 'contour area proportion of pic',
                default=0.1)
args = vars(ap.parse_args())

original = cv.imread(args['image'])
# 图像翻转，因为测试图片为白底，应该没有必要，后续优化掉
img = 255-original
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 添加高斯模糊，处理噪点，可能没有必要，但也许可以提高噪声图片寻找轮廓的速度
blur = cv.GaussianBlur(img_gray,(5,5),0)
# 二值化，用于寻找轮廓
ret,th = cv.threshold(blur,20,255,cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

# 计算图像面积，用于计算轮廓比例
pic_area = img.shape[0]*img.shape[1]
# 计算每一个轮廓的面积比例
cnt_area_list = [cv.contourArea(cnt)/pic_area for cnt in contours]
area_th = args['threshold']
# 根据阈值筛选轮廓
cnt_idx = np.where([c > area_th for c in cnt_area_list])[0]

# 选取轮廓用于抠图，后续应当改为可交互式操作选取
# cnt = contours[cnt_idx[-1]]




# 尝试添加鼠标点击获取轮廓序号
def get_coordinates(event,x,y,flags,param):
    global pt
    if event == cv.EVENT_LBUTTONDBLCLK:
        pt = (x,y)
        cv.destroyAllWindows()

cv.namedWindow('image')
cv.setMouseCallback('image',get_coordinates)
cv.imshow('image',original)
cv.waitKey(0)
cv.destroyAllWindows()

idx = np.where([cv.pointPolygonTest(contours[cnt_idx[i]],pt,False) == 1 for i in range(len(cnt_idx))])[0]
cnt = contours[cnt_idx[idx][0]]


# 使用轮廓生成mask
black_board = np.zeros(original.shape[:2])
fP = cv.fillPoly(black_board,[cnt],(255,255,255))
# 抠图并提取抠图区域
cut_out = cv.bitwise_and(original,original,mask=np.uint8(fP))
cut_out = cut_out[cnt[:,:,1].min():cnt[:,:,1].max(),
          cnt[:,:,0].min():cnt[:,:,0].max(),:]

cv.imshow('',cut_out)
cv.waitKey(0)


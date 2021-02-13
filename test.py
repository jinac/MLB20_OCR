import cv2
import numpy as np

img = cv2.imread('divleague2.png')
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)
h, w = img_bw.shape
# img_bw = cv2.medianBlur(img_bw, 5)

kernel = np.ones((3, 3), np.uint8)
# img_thresh = img_bw.
# img_thresh3 = cv2.Canny(img,50,200,3)
# ret, img_thresh1 = cv2.threshold(img_bw, 70, 250, cv2.THRESH_BINARY)
# ret, img_thresh2 = cv2.threshold(img_bw, 100, 255, cv2.THRESH_BINARY)
# img_thresh = cv2.adaptiveThreshold(img_bw, 210, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
ret, img_thresh1 = cv2.threshold(img_bw, 50, 255, cv2.THRESH_BINARY)
img_thresh1 = cv2.erode(img_thresh1, kernel, iterations=1)

ret, img_thresh2 = cv2.threshold(img_bw, 100, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_thresh2 = cv2.erode(img_thresh2, kernel, iterations=1)
img_thresh = img_thresh1 - img_thresh2
cv2.imwrite('th.jpg', img_thresh)
cv2.imwrite('th1.jpg', img_thresh1)
cv2.imwrite('th2.jpg', img_thresh2)

kernel = np.ones((3, 3), np.uint8)
img_erode = cv2.erode(img_thresh, kernel, iterations=1)
img_morph_clean = cv2.dilate(img_erode, kernel, iterations=1)

# ret, img_label = cv2.connectedComponents(img_thresh)
# labels = np.unique(img_label)
# boxes = []
# for label in labels:
#     y, x = np.where(img_label == label)  # row (y), col (x)
#     # print(y, x)
#     min_y = np.min(y)
#     min_x = np.min(x)
#     max_y = np.max(y)
#     max_x = np.max(x)
#     box_h_diff = np.abs(max_y - min_y)
#     box_w_diff = np.abs(max_x - min_x)
#     boxes.append([min_x, min_y, max_x, max_y])
#     print([min_x, min_y, max_x, max_y])
    

cv2.imwrite('thresh.jpg', img_morph_clean)
import cv2
import numpy as np


def gray(img):
	rows, cols, channels = img.shape
             
	result = img.copy()

	for i in range(rows):
        	for j in range(cols):
            		aux = 0
            		for k in range(channels):
                		aux += img[i, j, k]
                 
            		result[i, j] = aux / 3
     
    	return result


def CMY(img):
	rows, cols, channels = img.shape

	result = img.copy()

	for i in range(rows):
		for j in range(cols):
			for k in range(channels):
				result[i, j, k] = 255 - img[i, j, k]

	return result


def YCrCb(img):
	rows, cols, channels = img.shape

	result = img.copy()

	for i in range(rows):
		for j in range(cols):
			result[i, j, 0] = (0.299*img[i, j, 2]) + (0.587*img[i, j, 1]) + (0.114*img[i, j, 0])
            		result[i, j, 1] = (img[i, j, 2]-((0.299*img[i, j, 2]) + (0.587*img[i, j, 1]) + (0.114*img[i, j, 0])))*0.713 + 128
            		result[i, j, 2] = (img[i, j, 0]-(0.299*img[i, j, 2]) + (0.587*img[i, j, 1]) + (0.114*img[i, j, 0]))*0.713 + 128


	return result


def YUV(img):
	rows, cols, channels = img.shape

	result = img.copy()

	for i in range(rows):
		for j in range(cols):
			result[i, j, 0] = (0.299*img[i, j, 2]) + (0.587*img[i, j, 1]) + (0.114*img[i, j, 0])
            		result[i, j, 1] = (img[i, j, 2]-((0.299*img[i, j, 2]) + (0.587*img[i, j, 1]) + (0.114*img[i, j, 0])))
            		result[i, j, 2] = (img[i, j, 0]-(0.299*img[i, j, 2]) + (0.587*img[i, j, 1]) + (0.114*img[i, j, 0]))

	return result

img = cv2.imread("lena.jpg", 1)


gray = gray(img.copy())
cmy = CMY(img.copy())
ycrcb = YCrCb(img.copy())
yuv = YUV(img.copy())

cv2.imshow("Imagem", img)
cv2.imshow("Cinza", gray)
cv2.imshow("CMY", cmy)
cv2.imshow("YCrCb", ycrcb)
cv2.imshow("YUV", yuv)

cv2.waitKey(0)
cv2.destroyAllWindows()

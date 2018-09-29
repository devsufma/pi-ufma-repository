import numpy as np
import cv2, math


def add(img1, img2):
	
	rows, cols, channels = img1.shape

	add = np.zeros(rows*cols*channels, dtype = np.uint32).reshape((rows, cols, channels))
	
	for i in range(rows):
		for j in range(cols):
			for k in range(channels):
				if img1[i, j , k] + img2[i, j , k] > 255:
					add[i, j, k] = 255
				else:
					add[i, j, k] = img1[i, j , k] + img2[i, j , k]
	
	add = np.uint8(add)					 
	return add

def sub(img1, img2):
	
	rows, cols, channels = img1.shape

	sub = np.zeros(rows*cols*channels, dtype=np.int32).reshape((rows, cols, channels))
	
	for i in range(rows):
		for j in range(cols):
			for k in range(channels):
				if img1[i, j , k] - img2[i, j , k] < 0:
					sub[i, j, k] = 0
				else:
					sub[i, j, k] = img1[i, j , k] - img2[i, j , k]
	
	sub = np.uint8(sub)					 
	return sub

def mult(img1, fator):
	
	rows, cols, channels = img1.shape

	mult = np.zeros(rows*cols*channels, dtype=np.int32).reshape((rows, cols, channels))
	
	for i in range(rows):
		for j in range(cols):
			for k in range(channels):
				mult[i, j, k] = img1[i, j , k] * fator
	
	mult = np.uint8(mult)					 
	return mult

def div(img1, img2):
	
	rows, cols, channels = img1.shape

	div = np.zeros(rows*cols*channels, dtype=np.int32).reshape((rows, cols, channels))
	
	for i in range(rows):
		for j in range(cols):
			for k in range(channels):
				if img2[i, j , k] == 0:
					div[i, j, k] = 0
				else:
					div[i, j, k] = img1[i, j , k] / img2[i, j , k]
	
	div = np.uint8(div)					 
	return div

def mix(img1, alpha, img2, beta, gama):
	
	rows, cols, channels = img1.shape

	mix = np.zeros(rows*cols*channels, dtype=np.int32).reshape((rows, cols, channels))
	
	for i in range(rows):
		for j in range(cols):
			for k in range(channels):
				mix[i, j, k] = (alpha*img1[i, j , k])+(beta*img2[i, j , k])+gama
	
	mix = np.uint8(mix)				 
	return mix

def dist_euclidiana(img1, img2):
	
	tam1, tam2 = np.array(v1), np.array(v2)

	diff = tam1 - tam2
	
	quad_dist = np.dot(diff, diff)
					 
	return math.sqrt(quad_dist)
	


filename1 = "mario.jpg"
filename2 = "mega_mushroom.jpg"

img1 = cv2.imread(filename1, 1)
img2 = cv2.imread(filename2, 1)

f_img1 = np.int32(img1)
f_img2 = np.int32(img2)

img_add = add(f_img1, f_img2)
img_sub = sub(f_img1, f_img2)
img_mult = mult(img1, 0.5)
img_div = div(f_img1, f_img2)
img_mix = mix(f_img1, 0.6, f_img2, 0.4, 0)

cv2.imshow("Imagem 1", img1)
cv2.imshow("Imagem 2", img2)

cv2.imshow("ADD", img_add)
cv2.imshow("SUBTRACT", img_sub)
cv2.imshow("MULT", img_mult)
cv2.imshow("DIVIDE", img_div)
cv2.imshow("MIX", img_mix)


cv2.waitKey(0)
cv2.destroyAllWindows()

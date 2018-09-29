import numpy as np
import cv2,math

def mult_matriz(m1, m2):
	rows1, cols1 = m1.shape
	rows2, cols2 = m2.shape
	
	result = np.zeros(rows1*cols2, dtype=np.int32).reshape((rows1, cols2))

	if cols1 == rows2:
		for i in range(rows1):
			for j in range(cols2):
				aux = 0
				for k in range(rows2):
					aux += m1[i, k] * m2[k, j]
				result[i, j] = aux


	return result

def translation_img(img, x, y):
	
	rows, cols= img.shape

	m = np.float32([[1,0,x],[0,1,y]])

	size = rows*cols

	trans = np.zeros(size, dtype=np.int32).reshape((rows, cols))

	for i in range(rows-y):
		for j in range(cols-x):
			coord = np.float32([[j],[i],[1]])
			mult = mult_matriz(m, coord)
			trans[mult[1, 0], mult[0, 0]] = img[i, j] 
				
	trans = np.uint8(trans)					 
	return trans

def translation_img(img, x, y):
	
	rows, cols= img.shape

	m = np.float32([[1,0,x],[0,1,y]])

	size = rows*cols

	trans = np.zeros(size, dtype=np.int32).reshape((rows, cols))

	for i in range(rows-y):
		for j in range(cols-x):
			coord = np.float32([[j],[i],[1]])
			mult = mult_matriz(m, coord)
			trans[mult[1, 0], mult[0, 0]] = img[i, j] 
				
	trans = np.uint8(trans)					 
	return trans

def scale_img(img, fator):
	
	rows, cols= img.shape

	m = np.float32([[fator,0,0],[0,fator,0]])

	size = rows*fator*cols*fator

	scale = np.zeros(size, dtype=np.int32).reshape((rows*fator, cols*fator))

	for i in range(rows):
		for j in range(cols):
			coord = np.float32([[j],[i],[1]])
			mult = mult_matriz(m, coord)
			scale[mult[1, 0], mult[0, 0]] = img[i, j] 
				
	scale = np.uint8(scale)					 
	return scale


def rotation_img(img, angulo):
	
	rows, cols= img.shape

	rad = (angulo*3.14)/180

	m = np.float32([[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0]])

	size = rows*cols

	rotation = np.zeros(size, dtype=np.int32).reshape((rows, cols))

	for i in range(rows):
		for j in range(cols):
			coord = np.float32([[j],[i],[1]])
			mult = mult_matriz(m, coord)
			rotation[mult[1, 0], mult[0, 0]] = img[i, j] 
				
	

	rotation = np.uint8(rotation)					 
	return rotation

filename = "mario.jpg"

img = cv2.imread(filename, 0)

f_img = np.int32(img)

img_trans = translation_img(f_img, 100, 50)
img_scale = scale_img(f_img, 2)
img_rotation = rotation_img(f_img, 90)

cv2.imshow("Imagem", img)


#mostra imagens resultados
cv2.imshow("Translation", img_trans)
cv2.imshow("Scala", img_scale)
cv2.imshow("Rotation", img_rotation)


cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2, os, math

def quantizacao_uniforme_1(img, K):
	img = np.float32(img)
	quantized = img.copy()

	rows = img.shape[0]
	cols = img.shape[1]

	for i in range(rows):
		for j in range(cols):
			quantized[i,j] = (math.pow(2,K)-1) * np.float32((img[i,j] - img.min()) / (img.max() - img.min()))
			quantized[i,j] = np.round(quantized[i,j])*int(256/math.pow(2,K))

	return quantized

def quantizacao_uniforme_2(img, k):
	a = np.float32(img)
	bucket = 256/k
	quantizado = (a/bucket)
	return np.uint8(quantizado)*bucket

if __name__ == "__main__":
	
	filename = "mario.jpg"
	cores = [2, 8]
	for cor in cores:
		img = cv2.imread(filename, 0)
		resultado1 = quantizacao_uniforme_1(img, cor)
		resultado2 = quantizacao_uniforme_2(img, cor)

		name, extension = os.path.splitext(filename)
		new_filename1 = '{name}-quantizacao1-{k}-{ext}'.format(name=name, k=cor, ext=extension)
		new_filename2 = '{name}-quantizacao2-{k}-{ext}'.format(name=name, k=cor, ext=extension)
		cv2.imwrite(new_filename2, resultado2)
		cv2.imwrite(new_filename1, resultado1)
		
	

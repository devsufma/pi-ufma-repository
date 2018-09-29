import numpy as np
import cv2

def convolucao(img, threshold, mascara, windowsize):

	edge = windowsize//2

	value = 0

	rows, cols = img.shape

	new = np.zeros(rows*cols, dtype = np.float32).reshape((rows,cols))	

	for i in range(edge, rows-edge):
		for j in range(edge, cols-edge):
			
			value = 0

			for x in range(windowsize):
				for y in range(windowsize):

					value += img[i-edge+x,j-edge+y] * mascara[x][y]

			if(value > threshold):
				new[i,j] = 255
			else:
				new[i,j] = 0

	return new

def gaussiano(img):

	mascara = [1,2,1,2,4,2,1,2,1]
	windowsize = 3
	edge = windowsize//2
	value = 0

	neighbors = []

	rows, cols = img.shape

	new = np.zeros(rows*cols, dtype = np.float32).reshape((rows,cols))	

	for i in range(edge, rows-edge):
		for j in range(edge, cols-edge):
			
			for x in range(windowsize):
				for y in range(windowsize):
					neighbors.append(img[i-edge+x,j-edge+y])
			
			for k in range(len(neighbors)):
				value += neighbors[k] * mascara[k]

			result = (value/16)

			newpixel = np.round(result)

			if(newpixel < 0):
				newpixel = 0
			if(newpixel > 255):
				newpixel = 255

			new[i,j] = newpixel

			neighbors = []
			value = 0

	new = np.uint8(new)

	return new

def ponto(img, threshold):

	mascara = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
	windowsize = 3
	edge = windowsize//2

	value = 0

	rows, cols = img.shape

	new = np.zeros(rows*cols, dtype = np.float32).reshape((rows,cols))	

	for i in range(edge, rows-edge):
		for j in range(edge, cols-edge):
			
			value = 0

			for x in range(windowsize):
				for y in range(windowsize):

					value += img[i-edge+x,j-edge+y] * mascara[x][y]

			if(value > threshold):
				new[i,j] = 255
			else:
				new[i,j] = 0

	new = np.uint8(new)

	return new

def roberts(img, threshold):

	mascarax = [[1,0],[0,-1]]
	mascaray = [[0,-1],[1,0]]

	windowsize = 2
	edge = windowsize//2

	rows, cols = img.shape

	rbt = np.zeros(rows*cols, dtype = np.float32).reshape((rows,cols))

	rbtx = convolucao(img, threshold, mascarax, windowsize)
	rbty = convolucao(img, threshold, mascaray, windowsize)

	for i in range(edge, rows-edge):
		for j in range(edge, cols-edge):
			rbt[i,j] = abs(rbtx[i,j]) + abs(rbty[i,j])

	ret, roberts = cv2.threshold(rbt, 176, 255, cv2.THRESH_BINARY)

	return roberts

def prewitt(img, threshold):

	mascarax = [[-1,-1,-1],[0,0,0],[1,1,1]]
	mascaray = [[-1,0,1],[-1,0,1],[-1,0,1]]

	windowsize = 3
	edge = windowsize//2

	rows, cols = img.shape

	prwtt = np.zeros(rows*cols, dtype = np.float32).reshape((rows,cols))

	prwttx = convolucao(img, threshold, mascarax, windowsize)
	prwtty = convolucao(img, threshold, mascaray, windowsize)

	for i in range(edge, rows-edge):
		for j in range(edge, cols-edge):
			prwtt[i,j] = abs(prwttx[i,j]) + abs(prwtty[i,j])

	ret, prewitt = cv2.threshold(prwtt, 176, 255, cv2.THRESH_BINARY)

	return prewitt

def sobel(img, threshold):

	mascarax = [[-1,0,1],[-2,0,2],[-1,0,1]]
	mascaray = [[-1,-2,-1],[0,0,0],[1,2,1]]

	windowsize = 3
	edge = windowsize//2

	rows, cols = img.shape

	sbl = np.zeros(rows*cols, dtype = np.float32).reshape((rows,cols))

	sblx = convolucao(img, threshold, mascarax, windowsize)
	sbly = convolucao(img, threshold, mascaray, windowsize)

	for i in range(edge, rows-edge):
		for j in range(edge, cols-edge):
			sbl[i,j] = abs(sblx[i,j]) + abs(sbly[i,j])

	ret, sobel = cv2.threshold(sbl, 176, 255, cv2.THRESH_BINARY)

	return sobel

if __name__ == "__main__":

	horizontal = [[-1,-1,-1],[2,2,2],[-1,-1,-1]]
	mais45 = [[-1,-1,2],[-1,2,-1],[2,-1,-1]]
	vertical = [[-1,2,-1],[-1,2,-1],[-1,2,-1]]
	menos45 = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]

	filename = "lena.jpg"

	img = cv2.imread(filename, 0)

	f_img = np.float32(img)

	img_suave = gaussiano(f_img)

	descontinuidade_ponto = ponto(img, 100)	

	reta_horizontal = convolucao(img, 150, horizontal, 3)
	reta_mais45 = convolucao(img, 150, mais45, 3)
	reta_vertical = convolucao(img, 150, vertical, 3)
	reta_menos45 = convolucao(img, 150, menos45, 3)

	borda_roberts = roberts(img_suave, 20)
	borda_prewitt = prewitt(img_suave, 110)
	borda_sobel = sobel(img_suave, 130)

	cv2.imshow("Ponto", descontinuidade_ponto)

	cv2.imshow("Reta Horizontal", reta_horizontal)
	cv2.imshow("Reta +45", reta_mais45)
	cv2.imshow("Reta Vertical", reta_vertical)
	cv2.imshow("Reta -45", reta_menos45)

	cv2.imshow("Roberts", borda_roberts)
	cv2.imshow("Prewitt", borda_prewitt)	
	cv2.imshow("Sobel", borda_sobel)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

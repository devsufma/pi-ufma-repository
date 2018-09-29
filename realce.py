import cv2
import numpy

def negativo(img):
    rows, cols = img.shape
    
    result = img.copy()
    
    for i in range(rows):
        for j in range(cols):
            result[i, j] = 255 - img[i, j]
            
    return result 



def constraste(img, a, b, c, d):
    rows, cols = img.shape
    
    result = img.copy()
    
    for i in range(rows):
        for j in range(cols):
            result[i, j] = (img[i, j] - a) * ((d - c) / (b - a)) + c
            
            
    return result

def gama(img, y):
    rows, cols = img.shape
    
    result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))

    c = 1
    
    for i in range(rows):
        for j in range(cols):
            result[i,j] = c * numpy.power(img[i, j], y)
            
    return numpy.uint8(result)


def linear(img, g, d):
	rows, cols = img.shape

	result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))

	for i in range(rows):
		for j in range(cols):
			result[i, j] = g * img[i, j] + d

	return numpy.uint8(result)


def logaritmo(img):
	rows, cols = img.shape

	result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))

	g = 105.96

	for i in range(rows):
		for j in range(cols):
			result[i, j] = g * numpy.log10(img[i, j] + 1)


	return numpy.uint8(result)


def quadratic(img):
	rows, cols = img.shape
	
	result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))

	g = 1 / 255

	for i in range(rows):
		for j in range(cols):
			result[i, j] = g * numpy.power(img[i, j], 2)

	return numpy.uint8(result)


def raiz(img):
	rows, cols = img.shape

	result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))
	
	g = 255 / numpy.sqrt(255)
	
	for i in range(rows):
		for j in range(cols):	
			result[i, j] = g * numpy.sqrt(img[i, j])

	return numpy.uint8(result)


img = cv2.imread("lena.jpg", 0)

neg = negativo(img.copy())
con = constraste(img.copy(), 0, 255, 0, 255)
gam = gama(img.copy(), 2)
lin = linear(img.copy(), 1, 32)
log = logaritmo(img.copy())
qua = quadratic(img.copy())
raiz = raiz(img.copy())

cv2.imshow("Original", img)
cv2.imshow("Negativo", neg)
cv2.imshow("Constraste", con)
cv2.imshow("Gama", gam) 
cv2.imshow("Linear", lin)
cv2.imshow("Logaritmo", log)
cv2.imshow("Quadratico", qua)
cv2.imshow("Raiz", raiz)

cv2.waitKey(0)
cv2.destroyAllWindows()

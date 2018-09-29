import cv2
import numpy


def alogamento(img, phigh, plow):
	rows, cols = img.shape

	result = numpy.zeros((rows, cols), dtype=numpy.float32)

	level = numpy.power(2, 8) - 1

	maximum = numpy.max(img)

	minimum = numpy.min(img)

	for i in range(rows):
		for j in range(cols):
			if img[i, j] <= plow:
				result[i, j] = 0
			elif img[i, j] >= phigh:
				result[i, j] = 255
			else:
				result[i, j] = numpy.round(level * (img[i, j] - minimum) / (maximum - minimum))

	return numpy.uint8(result)


def equalize(img):
	img = numpy.float32(img)
	rows, cols = img.shape
	histogram = numpy.zeros(256)
	acumulateHistogram = numpy.zeros(256, dtype=numpy.float32)
#	factor = 255 / (rows * cols)
	factor = 0.00136
	
	result = numpy.zeros((rows, cols), dtype=numpy.float32)

	for i in range(rows):
		for j in range(cols):
			value = img[i, j]

			histogram[value] += 1

	acumulateHistogram = numpy.array(histogram)

		
	for i in range(256):
		if i == 0:
			acumulateHistogram[i] = histogram[i]
		else:
			acumulateHistogram[i] += acumulateHistogram[i - 1]


	for i in range(256):
		aux = acumulateHistogram[i] * factor
		
		acumulateHistogram[i] = round(aux)

	for i in range(rows):
		for j in range(cols):
			value = img[i, j]

			value = acumulateHistogram[value]

			result[i, j] = value 	

	return numpy.uint8(result)
	

img = cv2.imread("image.jpg", 0)

eq = equalize(img.copy())
al = alogamento(img.copy(), 200, 100)

cv2.imshow("Original", img)
cv2.imshow("Equalizado", eq)
cv2.imshow("Alongado", al)

cv2.waitKey(0)
cv2.destroyAllWindows()


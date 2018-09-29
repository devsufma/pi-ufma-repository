import cv2
import numpy

def basic(img):
	rows, cols = img.shape

	result = img.copy()

	threshold = (255 + 0) / 2
		
	for i in range(rows):
		for j in range(cols):
			if img[i, j] < threshold:
				result[i, j] = 0

			else:
				result[i, j] = 255

	return result 


def random(img):
	rows, cols = img.shape

	result = img.copy()

	threshold = (255 + 0) / 2

	for i in range(rows):
		for j in range(cols):
			tmp = img[i, j] + numpy.random.randint(-127, 128)

			if tmp < threshold:
				result[i, j] = 0
			
			else:
				result[i, j] = 255

	return result

def dp(img):

	img = numpy.float32(img)

	rows, cols = img.shape

	result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))

	dither = numpy.float32([[2, 3], [4, 1]])

	threshold = (255 + 0) / 2

	for i in range(rows):
		for j in range(cols):
			m = i % 2
			n  = j % 2

			if((img[i, j] / 255) > ( dither[m, n] / 5)):
				result[i, j] = 0
			else:
				result[i, j] = 255

	return numpy.uint8(result)


def floyd_steinberg(img):
	img = numpy.float32(img)

	rows, cols = img.shape

	result = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))
	simple = numpy.zeros(rows * cols, dtype=numpy.float32).reshape((rows, cols))
	threshold = (255 + 0) / 2

	for i in range(rows):
		for j in range(cols):
			if img[i, j] < threshold:
				simple[i, j] = 0
			else:
				simple[i, j] = 255

	for i in range(rows - 1):
		for j in range(cols - 1):

			error = img[i, j] - simple[i, j]

			result[i + 1, j] = img[i + 1, j] + (error * (7 / 16))
			result[i, j + 1] = img[i, j + 1] + (error * (5 / 16))
			result[i + 1, j + 1] = img[i + 1, j + 1] + (error * (1 / 16))
			result[i - 1, j + 1] = img[i - 1, j + 1] + (error * (3 / 16))

	return numpy.uint8(result)
	
filename = "lena.jpg"

img = cv2.imread(filename, 0)

basic = basic(img.copy())
random = random(img.copy())
dp = dp(img.copy())
fs = floyd_steinberg(img.copy())

cv2.imshow("Original", img)
cv2.imshow("Dithering Basico", basic)
cv2.imshow("Random", random)
cv2.imshow("Dispersao de Pontos", dp)
cv2.imshow("Floyd Steinberg", fs)

cv2.waitKey(0)
cv2.destroyAllWindows()

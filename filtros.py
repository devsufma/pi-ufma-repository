import cv2
import numpy

def media(img, length):
    value = 0
    edge = length // 2
    
    rows, cols = img.shape
    
    result = numpy.ones((rows, cols), dtype=numpy.float32)
    
    for i in range(rows - edge):
        for j in range(cols - edge):
            
            for x in range(length):
                for y in range(length):
                    value += img[i - edge + x, j - edge + y]
                    
            result[i, j] = numpy.round((value * 1) / (length * length))
            value = 0
            
    return numpy.uint8(result)

def gaussian(img):
    
    value = 0
    mask = numpy.array([1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    windowsize = 3
    edge = windowsize // 2
    
    neighbors = []
    
    rows, cols = img.shape
    
    result = numpy.ones((rows, cols), dtype=numpy.float32)
    
    for i in range(edge, rows - edge):
        for j in range(edge, cols - edge):
            for x in range(windowsize):
                for y in range(windowsize):
                    neighbors.append(img[i-edge+x, j-edge+y])
                    
            for k in range(len(neighbors)):
                value += neighbors[k] * mask[k]
                
            value = numpy.round(value / 16)
            
            if value < 0 :
                value = 0
            if value > 255:
                value = 255
                
            result[i, j] = value
            value = 0
            neighbors.clear()
            
    return numpy.uint8(result)

img = cv2.imread("lena.jpg", 0)

b = media(img, 5)
g = gaussian(img)

cv2.imshow("Original", img)
cv2.imshow("Filtro da Media", b)
cv2.imshow("Gaussiano", g)

cv2.waitKey(0)
cv2.destroyAllWindows()

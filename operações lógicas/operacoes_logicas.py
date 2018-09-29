import numpy as np
import cv2


def and_img(img1, img2):
    
    rows, cols = img1.shape

    img = np.zeros(rows*cols).reshape((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            if (img1[i, j] == 255 and img2[i, j] == 255):
                img[i, j] = 255
    
    return img

def or_img(img1, img2):
    
    rows, cols = img1.shape

    img = np.zeros(rows*cols).reshape((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            if (img1[i, j] == 255 or img2[i, j] == 255):
                img[i, j] = 255
    
    return img

def xor_img(img1, img2):
    
    rows, cols = img1.shape

    img = np.zeros(rows*cols).reshape((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            if (img1[i, j] == 255 and img2[i, j] != 255) or (img1[i, j] != 255 and img2[i, j] == 255):
                img[i, j] = 255
    
    return img

def not_img(img1):
    
    rows, cols = img1.shape

    img = np.zeros(rows*cols).reshape((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            if (img1[i, j] == 255):
                img[i, j] = 0
            else:
                img[i, j] = 255
                
    
    return img
    
filename1 = "1.png"
filename2 = "2.png"

img1 = cv2.imread(filename1, 0)
img2 = cv2.imread(filename2, 0)

img_and = and_img(img1, img2)
img_or = or_img(img1, img2)
img_xor = xor_img(img1, img2)
img_not1 = not_img(img1)
img_not2 = not_img(img2)

cv2.imshow("Imagem 1", img1)
cv2.imshow("Imagem 2", img2)
cv2.imshow("AND", img_and)
cv2.imshow("OR", img_or)
cv2.imshow("XOR", img_xor)
cv2.imshow("NOT Imagem 1", img_not1)
cv2.imshow("NOT Imagem 2", img_not2)

cv2.waitKey(0)
cv2.destroyAllWindows()

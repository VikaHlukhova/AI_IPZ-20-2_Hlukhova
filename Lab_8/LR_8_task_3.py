import cv2

img = cv2.imread("img.jpg")
print(img.shape)
imgResize = cv2.resize(img,(1000,500))
print(imgResize.shape)
imgCropped = img[40:300, 40:400]
cv2.imshow("Image",img)
cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)

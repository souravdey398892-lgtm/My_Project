#Steps
#1. Load images
#2. Display and covert to grayscale
#3. Load harrcascade classifier model for face detection
#4. Use function detectmultiscle() on grayscale image. It scans the images for faces
#5. Draw rectangles around detected faces
#6. Display the result using cv2.imshow() function
#7. Save result using cv2.imwrite() function
#8. Experiment with different photos. Adjust detection parameters (scaleFactor, minNeighbors)
#9. Document the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import cv2
#****************Load images******************#
image=cv2.imread(".\\Images\\Faces\\Face_5.jpg")
cv2.imshow("Face_image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#***********Convert to Grayscale**********#
Gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray_image", Gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#**********Load harrcascade classifier model for face detection*******#
print(cv2.data.haarcascades)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error")
else:
    print("model detected")

#*****Use function detectmultiscle() on grayscale image. It scans the images for faces****#
faces=face_cascade.detectMultiScale(Gray_image,scaleFactor=1.2,minNeighbors=5)
print(faces)

#******************Draw rectangle around detected faces*********************#
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Rectangle",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#************Save image *****************#
cv2.imwrite("Face_detected.jpg",image)
status=cv2.imwrite("Face_detected.jpg",image)
if status:
    print("Image saved")
else:
    print("Error in saving image")
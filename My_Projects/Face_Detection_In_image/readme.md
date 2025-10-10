🧠 Face Detection using OpenCV

📘 1. Project Overview



This mini-project detects human faces in images using OpenCV’s Haar Cascade Classifier.

It demonstrates fundamental computer vision skills — reading images, converting color spaces, and applying pretrained models for face detection.



🧩 2. Features



Detects one or multiple faces in an image



Draws bounding boxes around detected faces



Simple and lightweight implementation using OpenCV



Works with any image file (.jpg, .png, etc.)



🧰 3. Technologies Used



Python 🐍



OpenCV (cv2)



NumPy



📂 4. Dataset / Input



No dataset required.

You can use any image with one or more faces



⚙️ 5. Steps Performed



Load the image using OpenCV



Convert it from BGR to Grayscale



Load the Haar Cascade face detection model



Detect faces using detectMultiScale()



Draw rectangles around detected faces



Display the output image with detections



🧑‍💻 6. How to Run



Clone or download this repository



Install required libraries



pip install -r requirements.txt





Make sure your image is placed in a known folder (any path works)



Run the Python script:



python face\_detection.py





The detected faces will be displayed in a window.



📈 8. Example Output



📌 Detected faces will be marked with rectangles 



💬 9. Future Improvements



Extend to real-time detection using webcam (cv2.VideoCapture())



Add emotion recognition on detected faces



Use deep learning-based face detectors (like DNN or MTCNN)



✨ 10. Credits



Developed by Sourav Dey as part of Computer Vision learning.

Built using OpenCV and Python.


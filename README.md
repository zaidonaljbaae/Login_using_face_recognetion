# Face Recognition System
This personal project offers a hands-on solution for face recognition using Python. I utilized OpenCV for face detection and scikit-learn's KNeighborsClassifier for recognizing faces. The system is designed to provide both a practical tool for real-time face recognition and a learning experience in computer vision and machine learning.
# Key Components
Data Collection:

Script: add_faces.py
This script helps in gathering face images from a webcam. It prompts you to enter your name and then captures multiple images, which are essential for training the recognition model. To achieve better recognition results, it's recommended to run this script several times to collect a diverse set of images.
Face Recognition:

Script: main.py
This script handles real-time face recognition. It processes video from the webcam, detects faces, and uses a trained KNN classifier to identify them. Recognized faces are highlighted with a green rectangle, while unrecognized faces are marked with a red rectangle. Additionally, it shows a list of recognized names on the right side of the screen and can optionally record attendance.
# Key Features
Real-Time Detection: Detects and recognizes faces in a live video feed.
Visual Feedback: Uses colored rectangles to indicate recognized (green) and unrecognized (red) faces.
Customizable: Allows for easy adjustments and expansions, making it suitable for various applications like security systems or attendance tracking.
Interactive Learning: Provides hands-on experience with face recognition technologies and machine learning.
This project is both a practical tool and an educational opportunity, allowing you to explore the capabilities of face recognition technology while developing a useful application.

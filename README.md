# Sign_LAnguage_Detection
Sign Language Detection System

A real-time hand gesture recognition system that detects 20 sign language gestures using a custom-trained CNN model and live webcam feed.


Demo


📸  record some picture while running test.py
<img width="571" height="591" alt="image" src="https://github.com/user-attachments/assets/078bc852-d13a-4bfa-b82d-331df398ea64" />
<img width="601" height="484" alt="image" src="https://github.com/user-attachments/assets/ac58f6c1-38c3-4a67-ad8e-5f6c48ac7f84" />
<img width="647" height="417" alt="image" src="https://github.com/user-attachments/assets/5214d533-0abd-4453-a0d8-3dd89e3cd6a3" />


Recognized Signs (20 Classes)

Bye,Call,Good, Luck, Hello, I Love You, Left, Loser, Nice, No ,Not Okay, Ok Peace, Please, Power, Right, Rock, Smile, Stop, Thank You, Yes


How It Works


Webcam captures live video frames
cvzone HandDetector locates and crops the hand region
Cropped hand is resized to 224×224 on a white canvas (preserving aspect ratio)
Custom-trained Keras CNN classifies the gesture
Predicted label is displayed on screen in real time



Project Structure

sign-language-detection/
│
├── DataCollection.py     # Collect training images via webcam (press 's' to save)
├── test.py               # Run real-time detection via webcam (press 'q' to quit)
├── Data/                 # Collected images organized by class label
│   ├── Hello/
│   ├── Yes/
│   └── ...
├── Model/
│   ├── keras_model.h5    # Trained Keras model
│   └── labels.txt        # Class labels
└── README.md


Dataset


Collected by: Self-collected via webcam using DataCollection.py
Classes: 20 sign language gestures
Images per class: [ADD YOUR NUMBER — e.g. 300]
Total images: [ADD TOTAL — e.g. 6,000]
Preprocessing: Hand region cropped and padded to 224×224 on white background



Model


Framework: TensorFlow / Keras
Input size: 224 × 224 × 3 (RGB)
Classifier: cvzone ClassificationModule wrapping keras_model.h5
Output: Softmax over 20 classes
Test Accuracy: [ADD YOUR ACCURACY — e.g. 92%]



Installation & Setup

bashgit clone https://github.com/SarasvatiDevi/Sign_LAnguage_Detection
cd Sign_LAnguage_Detection
pip install opencv-python cvzone tensorflow numpy

To collect new training data:

bashpython DataCollection.py
# Press 's' to save each image
# Change the 'folder' variable in the script per class

To run real-time detection:

bashpython test.py
# Press 'q' to quit


⚠️ Update the model path in test.py to match your local directory before running.




Requirements

opencv-python
cvzone
tensorflow
numpy


Limitations


Trained on single-subject data — accuracy may vary across different hand sizes, skin tones, and lighting conditions
Indoor/controlled lighting only
One hand detected at a time (maxHands=1)
No mobile or web deployment yet



Future Work


Expand dataset to multiple subjects for better generalization
Add Pakistani Sign Language (PSL) specific gestures
Deploy as a web application using Flask or Streamlit
Extend to full sentence/phrase detection



Author

Sarasvati Devi

BS Artificial Intelligence — Aror University, Sukkur, Pakistan

LinkedIn · GitHub

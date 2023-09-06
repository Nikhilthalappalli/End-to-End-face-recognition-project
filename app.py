#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import numpy as np


face_recognition_model = load_model('vggface_model.h5')
mtcnn_detector = MTCNN()

labels = ['brad pitt', 'chris evans', 'tom criuse']

def recognize_face(face_crop):
    face_crop = cv2.resize(face_crop, (224, 224))
    face_crop = face_crop.astype('float32')
    face_crop /= 255.0
    face_embedding = face_recognition_model.predict(np.expand_dims(face_crop, axis=0))
    predicted_label_idx = np.argmax(face_embedding)
    if face_embedding[0][predicted_label_idx] < 0.9:
        predicted_label = 'Unknown'
    else:
        predicted_label = labels[predicted_label_idx]
    
    return predicted_label

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = mtcnn_detector.detect_faces(image_rgb)
    
    for face in faces:
        x, y, w, h = face['box']
        face_crop = image[y:y+h, x:x+w]
        recognized_label = recognize_face(face_crop)
        
        text_size = cv2.getTextSize(recognized_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        text_width, text_height = text_size
        text_x = max(x, 0) 
        text_y = max(y - 20, text_height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, recognized_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    image_tk = Image.fromarray(image)
    
    output_size = (600, 500)
    image_tk = image_tk.resize(output_size, Image.ANTIALIAS)
    
    image_tk = ImageTk.PhotoImage(image_tk)
    
    result_label.config(image=image_tk, width=output_size[0], height=output_size[1])
    result_label.image = image_tk


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        faces = mtcnn_detector.detect_faces(frame_rgb)
        
        for face in faces:
            x, y, w, h = face['box']
            face_crop = frame[y:y+h, x:x+w]
            
            recognized_label = recognize_face(face_crop)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, recognized_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        output_size = (500, 600)
        resized_frame = cv2.resize(frame, output_size)
        
        frame_tk = Image.fromarray(resized_frame)
        frame_tk = ImageTk.PhotoImage(frame_tk)
        result_label.config(image=frame_tk, width=output_size[0], height=output_size[1])
        result_label.image = frame_tk
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)


def choose_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_video(file_path)

root = tk.Tk()
root.title("Face Detection and Recognition")


image_button = tk.Button(root, text="Choose Image", command=choose_image)
video_button = tk.Button(root, text="Choose Video", command=choose_video)

result_label = tk.Label(root)

image_button.pack(pady=10)
video_button.pack(pady=10)
result_label.pack()

root.mainloop()


# In[ ]:





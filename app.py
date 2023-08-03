import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
import os
import subprocess

# Mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Parameters for pose estimation
pose_estimation_params = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Capture video from the webcam
cap = cv2.VideoCapture(1)  # change the index if necessary

# For logging
last_logged_time = time.time()
log_interval = 5 * 60  # 5 minutes

# For voice alert
last_alert_time = time.time()
alert_interval = 60  # 1 minute

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the BGR image to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Make detections
    results = pose_estimation_params.process(frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for the hip, shoulder, and neck
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        neck = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        
        # Calculate the angle in degrees
        shoulder_hip_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
        neck_shoulder_vector = np.array([neck[0] - shoulder[0], neck[1] - shoulder[1]])
        angle = np.degrees(np.arccos(np.dot(shoulder_hip_vector, neck_shoulder_vector) / 
                                     (np.linalg.norm(shoulder_hip_vector) * np.linalg.norm(neck_shoulder_vector))))
        
        # Log the angle every 5 minutes
        current_time = time.time()
        if current_time - last_logged_time > log_interval:
            with open('posture_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), angle])
            last_logged_time = current_time
        
        # Voice alert if the angle is too large (more than 35 degrees)
        if angle > 35 and current_time - last_alert_time > alert_interval:
            subprocess.call(['say', 'Bad posture, Bob'])
            last_alert_time = current_time
        
        # Draw a red frame around the image if the angle is too large (more than 35 degrees)
        if angle > 35:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
        
        # Draw the line along the back and the perpendicular line from the shoulder to the neck
        cv2.line(frame, (int(hip[0]*frame.shape[1]), int(hip[1]*frame.shape[0])), 
                 (int(shoulder[0]*frame.shape[1]), int(shoulder[1]*frame.shape[0])), (0, 255, 0), 2)
        cv2.line(frame, (int(shoulder[0]*frame.shape[1]), int(shoulder[1]*frame.shape[0])), 
                 (int(neck[0]*frame.shape[1]), int(neck[1]*frame.shape[0])), (0, 255, 0), 2)
        
        # Draw a dot at the shoulder
        cv2.circle(frame, (int(shoulder[0]*frame.shape[1]), int(shoulder[1]*frame.shape[0])), 5, (0, 255, 0), -1)
        
        # Draw a black belt under the text
        cv2.rectangle(frame, (10, 10), (600, 80), (0, 0, 0), -1)
        
        # Draw the angle on the frame with larger text
        cv2.putText(frame, f'Angle: {angle:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

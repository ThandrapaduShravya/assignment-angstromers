import cv2
import numpy as np
import time

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables to store tracking data
face_positions = []
eye_positions = []
timestamps = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (Haar Cascades work better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Get the current timestamp
    current_time = time.time()
    
    # Initialize variables to track face and eye detection status
    face_detected = False
    eyes_detected_properly = False
    
    # Loop over each detected face
    for (x, y, w, h) in faces:
        face_detected = True
        
        # Display "Face Detected" text
        cv2.putText(frame, 'Face Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Determine if the face is roughly centered
        frame_center_x = frame.shape[1] // 2
        face_center_x = x + w // 2
        
        if abs(frame_center_x - face_center_x) > w // 2:
            face_color = (0, 0, 255)  # Red if face is not centered
        else:
            face_color = (255, 255, 255)  # White if face is centered
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)
        
        # Store the face position and timestamp
        face_positions.append((x, y, w, h))
        timestamps.append(current_time)
        
        # Get the region of interest (ROI) in the grayscale frame for eye detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the ROI with adjusted parameters
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20), maxSize=(70, 70))
        
        # Check if eyes are detected
        if len(eyes) == 0:
            cv2.putText(frame, 'Error', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            eyes_detected_properly = False
        else:
            # Check if eyes are closed
            eyes_closed = False
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_aspect_ratio = (ew / float(eh))  # Aspect ratio of the eye
                
                # Define a threshold for eye aspect ratio to determine closed eyes
                if eye_aspect_ratio < 0.2:
                    eyes_closed = True
            
            if eyes_closed:
                cv2.putText(frame, 'Error: Eyes Closed', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                eyes_detected_properly = True
        
        # Initialize variables for eye center
        eye_centers = []
        
        # Loop over each detected eye
        for (ex, ey, ew, eh) in eyes:
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            eye_centers.append((eye_center_x, eye_center_y))
            
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        # Determine if the eyes are looking at the camera
        if len(eye_centers) == 2:
            left_eye, right_eye = sorted(eye_centers, key=lambda p: p[0])
            eye_distance = abs(left_eye[0] - right_eye[0])
            
            # Calculate average position of eyes
            avg_eye_x = (left_eye[0] + right_eye[0]) / 2
            
            # Determine if eyes are looking roughly at the center of the face
            if abs(avg_eye_x - face_center_x) < eye_distance / 2:
                gaze_color = (0, 255, 0)  # Green if eyes are looking at the camera
            else:
                gaze_color = (0, 0, 255)  # Red if eyes are not looking at the camera
            
            # Draw circles at eye centers
            for eye_center in eye_centers:
                cv2.circle(frame, eye_center, 5, gaze_color, -1)
        else:
            # If less than two eyes are detected
            cv2.putText(frame, 'Error', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # If no face is detected
    if not face_detected:
        cv2.putText(frame, 'Face Not Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame with the detected faces and eyes
    cv2.imshow('Face and Eye Tracking', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

# Save tracking data to a file
np.savez('tracking_data.npz', face_positions=face_positions, eye_positions=eye_positions, timestamps=timestamps)

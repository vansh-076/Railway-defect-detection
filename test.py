import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model=load_model("railway_track_classifier.h5")

vid=cv2.VideoCapture(r"C:\Users\kvans\OneDrive\Desktop\class 3\my project.mp4")
listpr=[]
while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    # Resize the frame to the input size of the model
    try:
        resized_frame = cv2.resize(frame, (256, 256))
    except cv2.error as e:
        print("Video ended or error in frame capture")
        print(listpr)
    # Normalize the image
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to match the input shape of the model
    input_frame = normalized_frame.reshape(1, 256, 256, 3)

    # Make predictions
    predictions = model.predict(input_frame)
    mp={
        0:"defective",
        1:"non-defective"
        }
    listpr.append(mp[int(np.where(predictions<0.5,0,1))])
    # Get the class with the highest probability
    predicted_class = int(np.where(predictions<0.5,0,1))

    # Display the predicted class on the frame
    cv2.putText(frame, f'Predicted Class: {mp[predicted_class]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
vid.release()
print(listpr)
cv2.destroyAllWindows()
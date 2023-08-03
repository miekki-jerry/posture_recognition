# Posture Alert Program
## Description
This program uses computer vision and machine learning techniques to detect and alert users about potential postural defects. 
It uses your webcam to capture real-time video, applies pose estimation to detect key landmarks on your body, and calculates the angle between your hip, shoulder, and neck. 
The program displays the calculated angle on the screen and visually marks the angle on your body. 
If the angle exceeds 35 degrees, indicating a potential postural defect, the program alerts you by drawing a red border around the screen and using Siri's voice to say "Bad posture, Bob".

This program is unique and cool because it:

- Helps to promote good posture habits by providing real-time feedback.
- Logs your posture data over time, allowing you to track your progress.
- Utilizes computer vision and machine learning techniques to analyze your posture.
  
### How to Install
To run the program on your machine, follow these steps:

1. Clone this repository to your local machine.

2. Install the necessary Python libraries. You can do this by running the following command in your terminal:

```
pip install opencv-python-headless mediapipe numpy
```

3. Run the program using Python:
   
```
python posture_alert.py
```

### Stack
This program is written in Python and uses the following libraries:

- OpenCV: Used for capturing video from the webcam, processing the frames, and displaying the results.
- MediaPipe: A machine learning library used for pose estimation.
- NumPy: Used for numerical operations, such as calculating the angle between vectors.
- Subprocess: Used for integrating with Siri to generate voice alerts.

Remember to adjust the camera index in the *_cv2.VideoCapture(1)* line to match your system configuration. The current index 1 is generally used for external cameras, while an index of 0 is used for built-in cameras.

Press 'q' to stop the program. This ensures that the video capture is properly released and that all windows are closed. The program logs the calculated angles in a CSV file named posture_data.csv in the same directory as the script. This file can be used to analyze your posture data over time.

If you encounter any issues while running the program, please check your system compatibility and ensure that all the necessary libraries are installed. This program has been tested on macOS and may require adjustments for other operating systems.

## Possible improvements
- Change voice from default macOS to Eleven Labs (maybe your mom's?) - https://elevenlabs.io/
- Add more detailed logs and create charts from this data
- Connect the program with room lights or with char electricity??

Team member names:
1.Kommineni Sai Sri Laxmi
2. vineetha

Liquid Level and Color Detection System

This project provides a liquid level and color detection system for saline bottles, leveraging computer vision to monitor liquid levels and color changes. The system issues alerts based on these detections, helping prevent empty or contaminated bottles in real-time scenarios, such as healthcare environments.

Features

Liquid Level Detection: Analyzes saline bottle images to determine the liquid fill percentage. Triggers alerts if the level drops below a specified threshold (e.g., 20%).

Color Detection: Detects changes in the color of the liquid, which can indicate potential contamination. Supports detection of several colors (e.g., Red, Yellow, Green, Blue, Clear).

Alerts: Plays a sound alert when low liquid level or color change is detected.

Real-Time Monitoring: Integrated with Roboflow's inference API for real-time analysis.

Technologies Used:

Python

OpenCV: For image processing and contour detection.

Pygame: For handling audio alerts.

Streamlit: For creating an interactive web application.

Roboflow: For object detection inference.

Installation

1.Clone the repository:
git clone https://github.com/yourusername/liquid-level-detection.git
cd liquid-level-detection

2.Install dependencies:
pip install -r requirements.txt

3.Set up Pygame Mixer:
Ensure Pygame is properly installed and initialize the mixer for audio alerts.

4.Configure Roboflow API:
Sign up at Roboflow and obtain an API key.
Update the api_key variable in the code with your Roboflow API key.

5.Place Alert Sounds:
Place alert sounds in the same directory or specify their paths in the code.
Usage
1.Run the Streamlit app:
streamlit run app.py
2.Upload an Image:
Upload an image of the saline bottle via the Streamlit interface.
The app will process the image to detect the liquid level and color, displaying alerts if necessary.

Code Overview

process_image: Main function to handle the uploaded image, perform inference via Roboflow, and annotate results.
preprocess_image: Prepares the image for contour analysis to detect the liquid level.
detect_liquid_level: Identifies the fill percentage and annotates the image with liquid level information.
detect_color: Detects the dominant color in the liquid region to identify potential contamination.
play_alert_sound: Plays an alert sound if a low-level or color change is detected.

Future Improvements
Expand color detection: Include additional colors to monitor for other types of contamination.
Enhance model accuracy: Refine the model on Roboflow for specific liquid detection.

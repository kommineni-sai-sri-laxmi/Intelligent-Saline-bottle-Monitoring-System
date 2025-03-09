import base64
import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
import pygame

# Initialize the client for Roboflow inference
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5Oijo8OWOGEpIsnA2iQz"  # Replace with your actual API key
)

# Initialize pygame mixer for alerts
pygame.mixer.init()

# Function to play an alert sound
def play_alert_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Function to preprocess the image and enhance contours
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned

# Function to detect color in the liquid region
def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'Red': ((0, 50, 50), (10, 255, 255)),
        'Yellow': ((15, 50, 50), (35, 255, 255)),
        'Green': ((36, 50, 50), (85, 255, 255)),
        'Blue': ((86, 50, 50), (125, 255, 255)),
        'Clear': ((0, 0, 200), (180, 30, 255))
    }
    max_pixels = 0
    detected_color = 'Unknown'
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_pixels = cv2.countNonZero(mask)
        if color_pixels > max_pixels:
            max_pixels = color_pixels
            detected_color = color
    return detected_color

# Function to detect and mark the liquid level
def detect_liquid_level(image, prediction):
    processed = preprocess_image(image)
    color_change_sound = "color_change_alert.wav"  # Replace with path to your color change sound
    low_level_sound = "low_level_alert.wav"        # Replace with path to your low-level alert sound

    # Get bounding box around the bottle
    x, y, w, h = int(prediction['x'] - prediction['width'] / 2), int(prediction['y'] - prediction['height'] / 2), int(prediction['width']), int(prediction['height'])
    roi_start_y = int(y + h * 0.2)
    roi_end_y = int(y + h * 0.8)
    liquid_level = None
    max_white_pixels = 0
    for i in range(roi_start_y, roi_end_y):
        row = processed[i, x:x + w]
        white_pixels = np.sum(row == 255)
        if white_pixels > max_white_pixels:
            max_white_pixels = white_pixels
            liquid_level = i
    if liquid_level is not None:
        cv2.line(image, (x, liquid_level), (x + w, liquid_level), (0, 0, 0), 2)
        fill_percentage = ((y + h - liquid_level) / h) * 100
        fill_percentage = min(100, max(0, fill_percentage))
        cv2.putText(image, f'Liquid Level: {fill_percentage:.1f}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect color in the liquid section
        liquid_roi = image[liquid_level:liquid_level + 10, x:x + w]
        detected_color = detect_color(liquid_roi)
        
        # Display detected color on the image
        cv2.putText(image, f'Color: {detected_color}', (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check for low-level alert
        if fill_percentage < 20:
            play_alert_sound(low_level_sound)  # Play sound for low liquid level

        return fill_percentage, detected_color
    return None, None

# Function to visualize the results
def visualize_results(image):
    st.image(image, channels="BGR")

# Main function to process the image
def process_image(uploaded_file):
    # Read the uploaded file as a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)  # Decode the image

    # Encode the image to base64 format for inference
    _, buffer = cv2.imencode('.jpg', original_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Perform inference using the Roboflow client
    result = CLIENT.infer(encoded_image, model_id="cv-90ctx/3")
    predictions = result['predictions']

    for prediction in predictions:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        width = int(prediction['width'])
        height = int(prediction['height'])
        confidence = prediction['confidence']

        # Draw bounding box and label on the image
        cv2.rectangle(original_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = f"{prediction['class']} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x
        text_y = y - 3 if y - 3 > 10 else y + 10
        cv2.rectangle(original_image, (text_x, text_y - text_size[1] - 1), 
                      (text_x + text_size[0], text_y + 1), (0, 0, 0), -1)
        cv2.putText(original_image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # Detect liquid level and color
        fill_percentage, detected_color = detect_liquid_level(original_image, prediction)
        if fill_percentage is not None:
            st.write(f"Detected fill percentage: {fill_percentage:.2f}%, Color: {detected_color}")

    # Visualize the processed image in Streamlit
    visualize_results(original_image)

# Streamlit app layout
st.title("Liquid Level and Color Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    process_image(uploaded_file)

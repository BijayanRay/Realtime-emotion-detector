import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, Response

# Initialize Flask app with custom template folder
app = Flask(__name__, template_folder='www')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
img_height, img_width = 48, 48
num_classes = 7

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * (img_height // 8) * (img_width // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * (img_height // 8) * (img_width // 8))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN().to(device)
model.load_state_dict(torch.load('emotion_recognition_model.pth', map_location=device))
model.eval()

# Emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Define image transformations for webcam input
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

def generate_frames():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Error: Could not open webcam.")

    # Define the custom boundary string
    boundary = b'--myboundary\r\n'

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale and apply transformations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(gray_frame)
        image = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        # Display the resulting frame with the predicted emotion
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Create the response header using the custom boundary
        header = (
            boundary +
            b'Content-Type: image/jpeg\r\n'
            b'\r\n'
        )

        # Yield the combined header and frame for streaming
        yield header + frame + b'\r\n'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=myboundary')

if __name__ == '__main__':
    app.run(debug=True)

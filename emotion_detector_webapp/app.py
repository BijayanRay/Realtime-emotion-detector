import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

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
try:
    model.load_state_dict(torch.load('emotion_detector_webapp/emotion_recognition_model.pth', map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Define image transformations for webcam input
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

st.title("Real-Time Emotion Detection")
run = st.checkbox("Run Real-Time Detection")

# Real-time processing loop
while run:
    # Capture webcam input using Streamlit's camera input
    img_input = st.camera_input("Webcam feed")

    if img_input:
        # Convert the captured image to grayscale
        img = Image.open(img_input).convert('L')  # Convert to grayscale
        image = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        # Display the resulting image with the predicted emotion
        st.image(img, caption=f"Predicted Emotion: {emotion}", use_column_width=True)
        time.sleep(0.1)  # Small delay to simulate real-time processing

    else:
        st.write("Waiting for webcam input...")

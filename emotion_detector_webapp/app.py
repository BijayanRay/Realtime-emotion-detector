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
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (img_height // 8) * (img_width // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Load the trained model once when the app starts
model = CNN().to(device)
model.load_state_dict(torch.load('emotion_detector_webapp/emotion_recognition_model.pth', map_location=device))
model.eval()

# Emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Define image transformations for webcam input
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

st.title("Real-Time Emotion Detection")

# Toggle button for running the detection
if st.button("Start" if not st.session_state.get("run", False) else "Stop"):
    st.session_state.run = not st.session_state.run

# Real-time processing logic
if st.session_state.get("run", False):
    # Capture webcam input using Streamlit's camera input
    img_input = st.camera_input("Webcam feed")

    if img_input:
        # Convert the captured image to grayscale
        img = Image.open(img_input).convert('L')  # Convert to grayscale
        image = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            predicted = output.argmax(dim=1).item()  # Get the predicted class index
            emotion = emotion_labels[predicted]

        # Display the predicted emotion
        st.markdown(f"**Predicted Emotion: {emotion}**")
        time.sleep(0.1)  # Small delay to prevent excessive updates
else:
    st.write("Click 'Start' to begin real-time detection.")

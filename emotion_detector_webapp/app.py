import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# st.write(f"Running on device: {device}")

# Define constants
img_height, img_width = 48, 48
num_classes = 7

# Define the CNN model with additional dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * (img_height // 8) * (img_width // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)  # Increased dropout to 0.5

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)  # Dropout after second pooling
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * (img_height // 8) * (img_width // 8))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model with additional dropout
model = CNN().to(device)
try:
    model.load_state_dict(torch.load('src/emotion_recognition_model.pth', map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Define image transformations for input
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

st.title("Emotion Detection")

# Initialize session state for tracking
if "run" not in st.session_state:
    st.session_state.run = False

# Function to toggle detection state
def toggle_detection():
    st.session_state.run = not st.session_state.run

# Toggle button for running the detection
st.button("Start" if not st.session_state.run else "Stop", on_click=toggle_detection)

# Emotion detection logic
def detect_emotion(image):
    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return emotion_labels[predicted.item()]

if st.session_state.run:
    # Capture webcam input using Streamlit's camera input
    img_input = st.camera_input("Webcam feed")

    if img_input:
        # Convert the captured image to grayscale
        img = Image.open(img_input).convert('L')  # Convert to grayscale
        image = transform(img).unsqueeze(0).to(device)

        # Get predicted emotion
        emotion = detect_emotion(image)

        # Display the predicted emotion
        st.markdown(f"**Predicted Emotion: {emotion}**")
else:
    # Show file uploader for image input when webcam is off
    img_upload = st.file_uploader("Upload an image to detect emotion", type=['jpg', 'jpeg', 'png'])
    
    if img_upload:
        # Display uploaded image as a preview
        img_preview = Image.open(img_upload)
        st.image(img_preview, caption="Uploaded Image", use_column_width=True)

        # Convert uploaded image to grayscale
        img = img_preview.convert('L')
        image = transform(img).unsqueeze(0).to(device)

        # Get predicted emotion
        emotion = detect_emotion(image)

        # Display the predicted emotion
        st.markdown(f"**Predicted Emotion: {emotion}**")

st.write("Click 'Start' to begin emotion detection via webcam or upload an image.")
st.markdown("Here is the GitHub repository [link](https://github.com/BijayanRay/realtime-emotion-detector) to this project.")

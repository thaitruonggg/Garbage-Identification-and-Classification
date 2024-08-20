import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Path to the classification model
CLASSIFICATION_MODEL_PATH = 'models/SwinTransformerModel.pth'  # Update with your classification model path

# Load the classification model
classification_model = torch.jit.load(CLASSIFICATION_MODEL_PATH)
classification_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model.to(device)

# Load a pre-trained object detection model (Faster R-CNN)
# Updated to use the new 'weights' parameter
detection_model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
detection_model.eval()
detection_model.to(device)

# Define the transformations to be applied to the image for detection
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define the transformations to be applied to the image for classification
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the expected input size
    transforms.ToTensor(),
])

# Initialize the video capture
cap = cv2.VideoCapture("external/v3.mp4")  # Video file
'''cap = cv2.VideoCapture(0)  # Webcam'''

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB for PIL and Torchvision
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame).convert('RGB')

    # Apply transformations and prepare input tensor
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Pass the image to the detection model to get the predictions
    with torch.no_grad():
        predictions = detection_model(input_tensor)

    # Extract bounding boxes, labels, and scores
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Set a threshold to filter out low-confidence detections
    threshold = 0.6
    filtered_indices = np.where(scores > threshold)[0]

    # Define class names
    class_names = ["battery", "biological", "cardboard", "glass", "metal", "paper", "plastic"]

    # Draw bounding boxes and annotate with classification and confidence score
    for idx in filtered_indices:
        box = boxes[idx]
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Crop the detected object and classify
        cropped_img = pil_image.crop((x_min, y_min, x_max, y_max))
        classification_input_tensor = classification_transform(cropped_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classification_model(classification_input_tensor)

        pred_class = torch.argmax(output, dim=1).item()
        confidence = scores[idx]
        label = f"{class_names[pred_class]}: {confidence:.2f}"

        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Output video', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

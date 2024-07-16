FILE = 'models/enhanced_model_1.pth'

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image
image_path = "external/cans.jpg"
image_name = image_path.split("/")[-1]
image = Image.open(image_path)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.jit.load(FILE)
model.eval()

# Define the transformations to be applied to the image
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Apply the transformations to the image
input_tensor = transform(image).unsqueeze(0)

# Pass the image to the model to get the predicted class
with torch.no_grad():
    output = model(input_tensor.to(device))

# Convert the output to a probability distribution
probs = torch.softmax(output, dim=1)

# Get the predicted class
pred_class = torch.argmax(probs).item()

# Define the class names
class_names = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper",
               "plastic", "shoes", "trash", "white-glass"]

# Print the predicted class and image name
print("The predicted class for the image", image_name, "is:", class_names[pred_class])
print("Processing via:", device)

# Display the image using Matplotlib
plt.imshow(image)
plt.title("Predicted Class: " + class_names[pred_class])
plt.show()

# Convert the PIL image to OpenCV format
img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Create a window to display the image
cv2.namedWindow("Predicted Class: " + class_names[pred_class])
cv2.imshow("Predicted Class: " + class_names[pred_class], img_cv)

# Wait for a key press
cv2.waitKey(0)

# Save the image to a new file
cv2.imwrite("output_image.jpg", img_cv)

# Close all OpenCV windows
cv2.destroyAllWindows()
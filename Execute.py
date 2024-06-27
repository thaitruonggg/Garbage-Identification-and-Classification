FILE = 'models/enhanced_model_1.pth'

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = "external/t1.jpg"
image_name = image_path.split("/")[-1]
image = Image.open(image_path)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.jit.load(FILE)
model.eval()

# Define the transformations to be applied to the image
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
])

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
class_names = ["battery", "biological", "brown-glass", "cardboard", "clothes", "glass", "green-glass", "metal", "paper",
               "plastic", "shoes", "trash", "white-glass"]

# Print the predicted class and image name
print("The predicted class for the image", image_name, "is:", class_names[pred_class])
print("Processing via:", device)

# Display the image
plt.imshow(image)
plt.title("Predicted Class: " + class_names[pred_class])
plt.show()
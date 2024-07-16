import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import multiprocessing

## DEFINING THE MODEL
def accuracy(outputs, labels):  # outputs is the output of the model, labels is the actual label of the image
    _, preds = torch.max(outputs, dim=1)  # preds is the predicted label of the image
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))  # returns the accuracy of the model

class ImageClassification(nn.Module):  # nn.Module is the base class for all neural network modules
    def training_step(self, batch):  # This function is used to train the model
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)
        return loss

    def validating(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'Validation Loss': loss.detach(), 'Validation Accuracy': acc}

    def validating_epoch_final(self, outputs):
        batch_loss = [x['Validation Loss'] for x in outputs]  # This line extracts the validation loss for
        # each batch of the validation data
        epoch_loss = torch.stack(batch_loss).mean()  # This line calculates the mean of the validation loss for all batches
        batch_accuracy = [x['Validation Accuracy'] for x in outputs]
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'Validation Loss': epoch_loss.item(), 'Validation Accuracy': epoch_accuracy.item()}

    def epoch_final(self, epoch, result):
        print("Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
              .format(epoch + 1, result['Training Loss'], result['Validation Loss'], result['Validation Accuracy']))

## USING ResNet50 FOR CLASSIFICATION
class ResNet(ImageClassification):  # Defining the ResNet Model
    def __init__(self):
        super().__init__()
        # Using ResNet50 pretrained model
        self.network = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        features = self.network.fc.in_features
        self.network.fc = nn.Linear(features, len(garbage_classes))  # Replacing last layer with a linear layer of
        # garbage classes with length 13

    def forward(self, image):
        return torch.sigmoid(self.network(image))  # Using sigmoid activation function

    def training_step(self, batch):
        images, labels = batch  # Get images and labels from batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def epoch_final(self, epoch, result):
        print("Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
              .format(epoch + 1, result['Train Loss'], result['Validation Loss'], result['Validation Accuracy']))

def main():
    ## LOADING THE DATASETS
    directory = 'all_classes/'
    global garbage_classes
    garbage_classes = os.listdir(directory)
    print(garbage_classes)

    ## TRANSFORMING THE DATASETS
    # Importing Transforms to modify the images
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms

    # Resize all images to 224x224 pixels and converting to Tensor
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Load the dataset and apply the transformations
    dataset = ImageFolder(directory, transform=transformations)

    ## TESTING THE DATASETS BY RANDOMLY DISPLAYING IMAGES AND THEIR LABELS (REMOVE ''' FROM LINE 87 & 94 TO MAKE IT WORK!)
    # Permute the dimensions of the image to fit the format of the matplotlib
    '''def display_test(image, label):
        print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

    # Display a random image from the dataset
    image, label = dataset[random.randint(0, len(dataset))]
    display_test(image, label)'''

    ## SETTING RANDOM SEED FOR REPRODUCIBILITY
    random_seed = 43
    torch.manual_seed(random_seed)

    ## SPLITTING DATASET & DEFINING BATCH SIZE
    # Split the dataset into train, validate, and test datasets
    train_size = int(0.6 * len(dataset))  # 60% of the dataset
    val_size = int(0.2 * len(dataset))  # 20% of the dataset
    test_size = len(dataset) - train_size - val_size  # Remaining 20% of the dataset
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    from torch.utils.data.dataloader import DataLoader
    batch_size = 32

    ## CREATING THE DATALOADERS
    train = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)

    ## BATCH VISUALIZATION
    from torchvision.utils import make_grid
    def batch_visualization(data):
        for image, labels in data:
            fig, ax = plt.subplots(figsize=(14, 14))
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.imshow(make_grid(image, nrow=16).permute(1, 2, 0))
            plt.show()
            break

    batch_visualization(train)
    model = ResNet()

    # Porting the model to GPU
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    device = get_default_device()

    # Move the model to GPU
    def move_to_gpu(data, device):
        # Move Tensor to GPU
        if isinstance(data, (list, tuple)):
            return [move_to_gpu(x, device) for x in data]
        return data.to(device, non_blocking=True)

    # DataLoading Class
    class DataLoad():
        def __init__(self, data, device):
            self.data = data
            self.device = get_default_device()

        # Yields batch after moving to device
        def __iter__(self):
            for batch in self.data:
                yield move_to_gpu(batch, self.device)

        # Returns the length of the data
        def __len__(self):
            return len(self.data)

    device = get_default_device()
    print('Processing via:', device)  # Note: If output is cuda, then GPU is available

    train = DataLoad(train, device)
    validation = DataLoad(validation, device)
    move_to_gpu(model, device)

    ## MODEL TRAINING
    @torch.no_grad()
    def evaluate(model, validator):
        model.eval()  # Evaluation Mode
        outputs = [model.validating(batch) for batch in validator]
        return model.validating_epoch_final(outputs)

    def opt(epochs, learning_rate, model, train_loader, validator, opt_func=torch.optim.SGD):
        training_history = []  # List to store the training history
        optimizer = opt_func(model.parameters(), learning_rate)
        for epoch in range(epochs):  # Loop for each epoch

            # Training Step
            model.train()  # Training Mode
            train_loss = []  # List to store the training loss
            for batch in train_loader:  # Loop for each batch
                loss = model.training_step(batch)  # Calculate the loss
                train_loss.append(loss)  # Append the loss to the list
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the parameters using the optimizer
                optimizer.zero_grad()  # Reset the gradients to zero for the next batch

            # Validation phase
            result = evaluate(model, validator)  # Evaluate the model on the validation set
            result['Train Loss'] = torch.stack(train_loss).mean().item()  # Calculate the average training loss
            model.epoch_final(epoch, result)  # Call the epoch_final method on the model
            training_history.append(result)  # Append the results to the training history
        return training_history

    model = move_to_gpu(ResNet(), device)
    evaluate(model, validation)

    ## DEFINING HYPERPARAMETERS
    epoch = 10
    optimizer = torch.optim.Adam
    learning_rate = 0.00005
    model_history = opt(epoch, learning_rate, model, train, validation, optimizer)

    ## PLOTTING THE ACCURACY VS EPOCHS
    def plot_accuracy(model_history):
        accuracies = [x['Validation Accuracy'] for x in model_history]
        plt.plot(accuracies, '-x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.show()
    plot_accuracy(model_history)

    def plot_loss(model_history):
        train_loss = [x.get('Train Loss') for x in model_history]
        validation_loss = [x['Validation Loss'] for x in model_history]
        plt.plot(train_loss, '-bx')
        plt.plot(validation_loss, '-rx')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. Epochs')
        plt.show()
    plot_loss(model_history)

    ## VISUALIZE THE PREDICTIONS
    def predict(image, model):
        # Convert to a batch of 1
        xb = move_to_gpu(image.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        return dataset.classes[preds[0].item()]

    ### TESTING NO.1
    img, label = random.choice(dataset)
    plt.imshow(img.permute(1, 2, 0))  # Permuting the image to the format expected by matplotlib
    print('Testing No.1 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
    plt.title('Testing No.1')
    plt.show()

    ### TESTING NO.2
    img, label = random.choice(dataset)
    plt.imshow(img.permute(1, 2, 0))  # Permuting the image to the format expected by matplotlib
    print('Testing No.2 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
    plt.title('Testing No.2')
    plt.show()

    ### TESTING NO.3
    img, label = random.choice(dataset)
    plt.imshow(img.permute(1, 2, 0))  # Permuting the image to the format expected by matplotlib
    print('Testing No.3 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
    plt.title('Testing No.3')
    plt.show()

    ## PREDICTING ON A RANDOM IMAGE
    loaded_model = model
    from PIL import Image  # For loading images
    from pathlib import Path  # For getting the path of the image
    def predict_random(image_name):  # Function to predict the class of a random image
        image = Image.open(Path('external' + '/' + image_name))
        example = transformations(image)  # Transforming the image to the format expected by the model
        plt.imshow(example.permute(1, 2, 0))
        print('Predicted Class:', predict(example, loaded_model))
        plt.title('Testing Random Image')
        plt.show()
    predict_random('cans.jpg')

    # SAVING THE MODEL
    FILE = "models/enhanced_model_1.pth"
    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, FILE)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
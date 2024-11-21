import torch  # Imports the PyTorch library, which is fundamental for building neural networks.
import torchvision.transforms as transforms  # Imports the transforms module from torchvision for preprocessing images.
from torch.utils.data import DataLoader, Dataset # Imports DataLoader for batching data and Dataset as a base class for datasets.
from torchvision import datasets  # Imports the datasets module from torchvision to handle common datasets.
import torch.nn as nn  # Imports the neural networks module, providing the building blocks for neural networks.
import torch.optim as optim  # Imports the optim module to provide common optimization algorithms like SGD, Adam.
from PIL import Image  # Imports the Python Imaging Library module to handle image loading and manipulation.
import os  # Imports the os module, useful for interacting with the operating system.

# Define the dataset class
class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index] # Retrieves the path and target of the image at the given index.
        img = self.loader(path)  # Loads the image using the defined loader method.
        img_resized = img.resize((128, 128)) # Resizes the image to 128x128 pixels.
        img_gray = img_resized.convert('L')  # Converts the resized image to grayscale.
        img_gray = transforms.ToTensor()(img_gray) # Converts the grayscale image to a PyTorch tensor.
        img_color = transforms.ToTensor()(img_resized) # Converts the resized color image to a PyTorch tensor.
        return img_gray, img_color # Returns the grayscale tensor and the color tensor.

# Load data
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])  # Compose a series of transformations.
train_dataset = GrayscaleImageFolder(root='C:\\DeepLearning\\Assignment3\\Assign3\\training_data', transform=transform) # Loads the dataset.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Creates a DataLoader to batch and shuffle the dataset.

# Define the model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__() # Initializes the superclass.
        self.encoder = nn.Sequential(  # Defines the encoder part of the network using sequential layers.
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # First convolution layer (grayscale input).
            nn.ReLU(), # Activation function.
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Second convolution layer
            nn.ReLU(), # Activation function.
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Third convolution layer
        )
        self.decoder = nn.Sequential(  # Defines the decoder part of the network using sequential layers.
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # First deconvolution layer.
            nn.ReLU(), # Activation function.
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Second deconvolution layer
            nn.ReLU(), # Activation function.
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1), # Last layer to produce a 3-channel output.
            nn.Sigmoid()  # Ensures output pixel values are in the range [0, 1].
        )

    def forward(self, x):
        x = self.encoder(x) # Passes the input through the encoder.
        x = self.decoder(x) # Passes the encoded features through the decoder.
        return x # Returns the decoded (colorized) image.

model = ColorizationNet()  # Instantiates the model.
criterion = nn.MSELoss()   # Sets Mean Squared Error as the loss function.
optimizer = optim.Adam(model.parameters(), lr=0.001) # Sets Adam as the optimizer with a learning rate of 0.001.

# Training loop
def train(model, dataloader, epochs=4):
    model.train() # Sets the model to training mode.
    for epoch in range(epochs): # Iterates over each epoch
        for data, target in dataloader: # Iterates over each batch of data.
            optimizer.zero_grad() # Clears the gradients of all optimized tensors.
            output = model(data)  # Feeds the input data through the model.
            loss = criterion(output, target)  # Computes the loss.
            loss.backward() # Backpropagates the error.
            optimizer.step() # Updates the model parameters.
        print(f'Epoch {epoch+1}, Loss: {loss.item()}') # Prints the loss at the end of each epoch.

train(model, train_loader) # Calls the training function.

# Save the model
torch.save(model.state_dict(), './colorizing_2859145.pth') # Saves the trained model parameters.

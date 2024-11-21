import torch # Imports PyTorch, a deep learning library.
import torch.nn as nn # Imports the neural network components from PyTorch.
import torchvision.transforms as transforms # Provides common image transformations.
from PIL import Image # Used for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt  # Used for plotting, here to display images.
import os # Provides a way of using operating system dependent functionality like reading or writing to a file.

# Define the model architecture
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__() # Calls the constructor of the parent class nn.Module.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # First convolution layer, for feature extraction.
            nn.ReLU(), # ReLU activation function, adds non-linearity.
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Second convolution layer
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Third convolution layer
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Decoder begins, upsamples.
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Continues to upsample.
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1), # Final layer to produce 3-channel output.
            nn.Sigmoid()  # Ensures output values are between 0 and 1, suitable for image data.
        )

    def forward(self, x):
        x = self.encoder(x) # Process input through the encoder.
        x = self.decoder(x) # Process input through the encoder.
        return x # Returns the colorized image.

# Load the trained model
model = ColorizationNet() # Instantiate the model.
model.load_state_dict(torch.load('./colorizing_2859145.pth'))  # Loads the trained model weights.
model.eval()  # Sets the model to evaluation mode (affects Dropout and BatchNorm).

# Transform function to convert image to tensor and resize
def to_tensor_and_resize(image_path):
    img = Image.open(image_path).convert('L') # Open and convert the image to grayscale.
    img = img.resize((128, 128))  # Resize to match the expected input dimensions
    img = transforms.ToTensor()(img).unsqueeze(0) # Convert the image to a tensor and add a batch dimension.
    return img

# Function to process and colorize images from a directory
def colorize(directory_path): 
    for filename in os.listdir(directory_path): # Iterate over files in the directory.
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Filter for image files.
            file_path = os.path.join(directory_path, filename) # Construct the full path to the file.
            input_image = to_tensor_and_resize(file_path) # Process the image.
            with torch.no_grad(): # Disable gradient computation.
                output = model(input_image) # Get the colorized image from the model.
            output_image = output.squeeze().permute(1, 2, 0).numpy() # Reformat the tensor for plotting.
            plt.imshow(output_image) # Display the image using matplotlib.
            plt.title(f'Colorized Image: {filename}') # Set the title of the plot.
            plt.show() # Show the plot.

# Usage example, provide the path to your directory of grayscale images
colorize('C:\\DeepLearning\\Assignment3\\Assign3\\testing_data') # Call the colorize function on a specified directory.

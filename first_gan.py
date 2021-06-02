import torch
from IPython import display
from torch import nn, optim
import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import pdb
from utils import Logger
from math import *

real_images_folder = './monet_jpg'

#Write Custom Pytorch Dataset Class

class monet_dataset(Dataset):
    """
    Monet pictures dataset
    """

    def __init__(self,path):

        self.path = path
        self.elements_to_dict ={}

        for index,file in enumerate(os.listdir(self.path)):
            self.elements_to_dict[index] = file


    def __len__(self):

        return(len(os.listdir(self.path)))

    def __getitem__(self,idx):

        img = cv2.imread(os.path.join(self.path,self.elements_to_dict[idx]))
        trans = transforms.ToTensor()
        tensor = trans(img)
        return tensor

        

data = monet_dataset(path = real_images_folder)

data_loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)

num_batches = len(data_loader)



class DiscriminatorNet_old(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 3*256*256
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        pdb.set_trace()
        return x

class DiscriminatorNet(torch.nn.Module):
    """
    Using CNN to build discriminaorNet
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 3*256*256
        h_in = 256,
        w_in = 256
        n_out = 1

        self.conv1 = nn.Conv2d(
                in_channels = 3, out_channels = 30, kernel_size= 7, stride = 1,padding = 0)

        self.bc1 = nn.BatchNorm2d(30)

        """
        
        Formula for calculating the Output height and width after the tensor passes through a kernel

        H_out = floor((H_in + 2*padding - dilation(kernel_size - 1)-1)/stride +1)
        W_out = floor((W_in + 2*padding - dilation(kernel_size - 1)-1)/stride +1)

        

        h_out_1 = floor( h_in + 2*0 - 1*(7-1)-1  + 1 )

        w_out_1 = floor( w_in + 2*0 - 1*(7-1)-1  + 1 )

        """

        self.relu = nn.LeakyReLU(0.2)

        self.max_pool = nn.MaxPool2d(kernel_size = 2, padding = 0)


        self.conv2 = nn.Conv2d(
            in_channels = 30, out_channels=100, kernel_size = 5, stride = 1, padding = 0)

        self.bc2 = nn.BatchNorm2d(100)

        self.conv3 = nn.Conv2d(
            in_channels = 100, out_channels = 30,kernel_size = 3,stride = 1, padding = 0)

        self.bc3 = nn.BatchNorm2d(30)

        self.conv4 = nn.Conv2d(
            in_channels = 30, out_channels = 10, kernel_size = 5, stride = 1, padding = 0)

        self.bc4 = nn.BatchNorm2d(10)

        #self.flatten = torch.flatten()

        self.lin_layer1 = nn.Linear(1440,1000)

        self.lin_layer2 = nn.Linear(1000,250)

        self.lin_layer3 = nn.Linear(250,50)

        self.lin_layer4 = nn.Linear(50,1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bc1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.bc2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.bc3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.bc4(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = torch.flatten(x,start_dim=1)

        x = self.lin_layer1(x)
        x = self.relu(x)

        x = self.lin_layer2(x)
        x = self.relu(x)

        x = self.lin_layer3(x)
        x = self.relu(x)

        x = self.lin_layer4(x)
        x = self.relu(x)
        x = self.sigmoid(x)

        return x

    
def images_to_vectors(images):
    return images
    #return images.view(images.size(0),3*256*256)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 3, 256, 256)


#Generator Network

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 3*256*256
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        x = torch.reshape(x,(x.shape[0],3,256,256))
        return x
    
# Noise
def noise(size):
    n = Variable(torch.randn(size,100))
    if torch.cuda.is_available(): return n.cuda() 
    return n

discriminator = DiscriminatorNet()
generator = GeneratorNet()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()



# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.002)

# Loss function
loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  
# Number of epochs
num_epochs = 20

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data



def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


num_test_samples = 16
test_noise = noise(num_test_samples)



logger = Logger(model_name='VGAN', data_name='monet_dataset')

for epoch in range(num_epochs):
    for n_batch, real_batch in enumerate(data_loader):

        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        if torch.cuda.is_available(): real_data = real_data.cuda()

        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()

        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)



        # Display Progress
        if (n_batch) % 5 == 0:
            display.clear_output(True)
            # Display Images
            test_images = vectors_to_images(generator(test_noise)).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)
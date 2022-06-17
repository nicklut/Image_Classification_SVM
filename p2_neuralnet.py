!pip install torch

!pip install torchvision

!pip install pytorch-ignite

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import math
import random
import sys
import copy
import glob
import cv2

from google.colab import drive
drive.mount('/content/gdrive')

!unzip gdrive/My\ Drive/hw6_data.zip

def read_images(folder):
  old_dir = os.getcwd()
  new_dir = os.path.join(old_dir, folder)

  img_list = []
  img_names = []
  img_type = [] 
  
  img_classifications = ["grass", "ocean", "redcarpet", "road", "wheatfield"]
  img_class = [0, 1, 2, 3, 4]
  img_size = 25

  for i in range(len(img_classifications)):
    temp_dir = new_dir
    new_dir = os.path.join(new_dir, img_classifications[i])

    for filename in glob.glob(new_dir + '/*.JPEG'):
      img = cv2.imread(filename)
      resized = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
      img_list.append(resized)
      img_names.append(filename)
      temp = np.zeros(5, dtype=int)
      temp[i] = 1
      img_type.append(temp)
    
    new_dir = temp_dir
  
  os.chdir(old_dir)
  img_list = torch.from_numpy(np.array(img_list))
  img_list = img_list.permute(0,3,1,2)
  return img_list, torch.from_numpy(np.array(img_type)), np.array(img_names)

  # Read in training, validation, and training data
# WE ARE NOT USING THE DESCRIPTORS, the inputs should directly be images
# 1) Downsize the images 
# 2) Flatten the images into data x that is the input to the neural network

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_x, train_y, train_names = read_images("train")
test_x, test_y, test_names = read_images("test")
valid_x, valid_y, vaild_names = read_images("valid")
point_dim = len(train_x[0])

# Create dataset class 
class imageDataset(Dataset):
  def __init__(self, data_x, data_y):
    self.images = data_x
    self.labels = data_y
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    sample = {'image': self.images[idx], 'label': self.labels[idx]}

    return sample

import torch.nn as nn
import torch.nn.functional as F
# Where point_dim is the dimension of the descriptor 
# and Nout is 5 since we are classifying between 5 types of backgrounds
N0 = train_x.shape[1]*train_x.shape[2]*train_x.shape[3]
N1 = 1500
N2 = 1500
Nout = 5

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Create three fully connected layers, two of which are hidden and the third is
        # the output layer.  In each case, the first argument to Linear is the number
        # of input values from the previous layer, and the second argument is the number
        # of nodes in this layer.  The call to the Linear initializer creates a PyTorch
        # functional that in turn adds a weight matrix and a bias vector to the list of
        # (learnable) parameters stored with each Net object.  These weight matrices
        # and bias vectors are implicitly initialized using a uniform random distribution,
        # in the range [-1/sqrt(k), 1/sqrt(k)] where k is the number of units at the
        # previous layer.
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N0, N1),
            nn.ReLU(),
            nn.Linear(N1, N2),
            nn.ReLU(),
            nn.Linear(N2, Nout)
        )

        # self.fc1 = nn.Linear(N0, N1, bias=True)
        # self.fc2 = nn.Linear(N1, N2, bias=True)
        # self.fc3 = nn.Linear(N2, Nout, bias=True)

    def forward(self, x):
        #  The forward method takes an input Variable and creates a chain of Variables
        #  from the layers of the network defined in the initializer. The F.relu is
        #  a functional implementing the Rectified Linear activation function.
        #  Notice that the output layer does not include the activation function.
        #  As we will see, that is combined into the criterion for the loss function.
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x # logit 

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

     
class NeuralNetwork_Conv(nn.Module):
  def __init__(self):
    super(NeuralNetwork_Conv, self).__init__()
    self.conv_stack = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2,2),
           #nn.Conv2d(16, 32, 3, stride=1, padding=1),
           #nn.ReLU(),
           #nn.MaxPool2d(2,2),
           nn.Conv2d(16, 32, 3, stride=1, padding=1),
           nn.ReLU(),
    )

    # Resulting image should be 7x7x32 
    self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
    )

  def forward(self, x):
    logits = self.fc_stack(self.conv_stack(x))
    return logits


#  Create an instance of this network.
model = Net().to(device)

#  Define the Mean Squared error loss function as the criterion for this network's training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#  Print a summary of the network.  Notice that this only shows the layers
print(model)

#  Print a list of the parameter sizes
params = list(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()

  for batch, sample_batched in enumerate(dataloader):
    X = sample_batched['image'].to(device)
    y = sample_batched['label'].to(device)

    # Compute the prediction error
    pred = model(X.float())
    loss = loss_fn(pred, y.float())

    # Backpropagation 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

def test(data, model, loss_fn, batch_size, prev_correct, prev_model):
    size = len(dataloader.dataset)/batch_size
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    count = 0
    predictions = np.array([])
    actual = np.array([])
    with torch.no_grad():
        for batch, sample in enumerate(data):
            X, y = sample['image'].to(device), sample['label'].to(device)
            pred = model(X.float())
            #print("Pred",pred.argmax(1))
            #print("y:", y)
            #print(y.argmax(1))
            temp1 = pred.argmax(1).cpu().numpy()
            temp2 = y.argmax(1).cpu().numpy()
            correct += np.sum(np.equal(temp1, temp2))
            predictions = np.concatenate((predictions, temp1))
            actual = np.concatenate((actual, temp2)) 

            test_loss += loss_fn(pred, y.float()).item()
            count += 1
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    
    correct /= (count*batch_size)
    confusion = confusion_matrix(actual, predictions)
    #print(size)
    
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(confusion)
    if correct > prev_correct:
      return correct, model
    else:
      return prev_correct, prev_model 

epochs = 10
batch_size = 5
learning_rate = 1e-4 

train_x = train_x.to(device)
train_y = train_y.to(device)
valid_x = valid_x.to(device)
valid_y = valid_y.to(device)

img_dataset = imageDataset(data_x = train_x, data_y = train_y)
dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = imageDataset(data_x = valid_x, data_y = valid_y)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

#train(dataloader, model, loss_fn, optimizer)

for t in range(epochs):
  train(dataloader, model, loss_fn, optimizer)
  
  
print("Training Data Results:")
correct, model = test(valid_dataloader, model, loss_fn, batch_size, 0, 0)

valid_x = valid_x.to(device)
valid_y = valid_y.to(device)
valid_dataset = imageDataset(data_x = valid_x, data_y = valid_y)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
final_e = epochs
final_lr = 1e-3
final_model = model
final_correct = correct
# Now lets run validation 
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
epochs_list = [10, 50, 100]
#epochs_list = [1, 10]
#  Create an instance of this network.

for rate in learning_rates: 
  for epochs in epochs_list:
    model = Net().to(device)

    #  Define the Mean Squared error loss function as the criterion for this network's training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)

    for t in range(epochs):
      train(dataloader, model, loss_fn, optimizer)
    
    prev_correct = correct
    print("Results for lr:", rate, "epochs:", epochs)
    correct, model = test(valid_dataloader, model, loss_fn, batch_size, prev_correct, model)
    
    if correct > prev_correct: 
      final_e = epochs 
      final_lr = rate
      final_model = model
      final_correct = correct

print("Final model has", final_e, "epochs and a learning rate of", final_lr, "with validation accuracy of", final_correct)
print()
print("Final test results are:")

test_x = test_x.to(device)
test_y = test_y.to(device)
test_dataset = imageDataset(data_x = test_x, data_y = test_y)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

test(test_dataloader, final_model, loss_fn, batch_size, prev_correct, model)

# Create convolutional neural network 
#  Create an instance of this network.
model = NeuralNetwork_Conv().to(device)

#  Define the Mean Squared error loss function as the criterion for this network's training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
batch_size = 5
learning_rate = 1e-4 

train_x = train_x.to(device)
train_y = train_y.to(device)
valid_x = valid_x.to(device)
valid_y = valid_y.to(device)

img_dataset = imageDataset(data_x = train_x, data_y = train_y)
dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = imageDataset(data_x = valid_x, data_y = valid_y)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

#train(dataloader, model, loss_fn, optimizer)

for t in range(epochs):
  train(dataloader, model, loss_fn, optimizer)
  
print("Convolutional Training Data Results:")
correct, model = test(valid_dataloader, model, loss_fn, batch_size, 0, 0)

valid_x = valid_x.to(device)
valid_y = valid_y.to(device)
valid_dataset = imageDataset(data_x = valid_x, data_y = valid_y)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
final_e = epochs
final_lr = 1e-3
final_model = model
# Now lets run validation 
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
epochs_list = [10, 50, 100]
#epochs_list = [1, 10]
#  Create an instance of this network.

for rate in learning_rates: 
  for epochs in epochs_list:
    model = Net().to(device)

    #  Define the Mean Squared error loss function as the criterion for this network's training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)

    for t in range(epochs):
      train(dataloader, model, loss_fn, optimizer)
    
    prev_correct = correct
    print("Results for lr:", rate, "epochs:", epochs)
    correct, model = test(valid_dataloader, model, loss_fn, batch_size, prev_correct, model)
    
    if correct > prev_correct: 
      final_e = epochs 
      final_lr = rate
      final_model = model

print("Final convolutional model has", final_e, "epochs and a learning rate of", final_lr)
print()
print("Final test results are:")

test_x = test_x.to(device)
test_y = test_y.to(device)
test_dataset = imageDataset(data_x = test_x, data_y = test_y)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test(test_dataloader, final_model, loss_fn, batch_size, prev_correct, model)


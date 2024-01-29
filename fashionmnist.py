import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''


'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''

# Given...
# transform = transforms.Compose(''' Fill in this function ''')  # Use transforms to convert images to tensors and normalize them
# batch_size = ''' Insert a good batch size number here '''

# normalizing tensor values
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
batch_size = 64


'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

# Given...
# trainset = torchvision.datasets.FashionMNIST(''' Fill in this function ''')
# trainloader = torch.utils.data.DataLoader(''' Fill in this function ''')

# testset = torchvision.datasets.FashionMNIST(''' Fill in this function ''')
# testloader = torch.utils.data.DataLoader(''' Fill in this function ''')

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

'''
PART 3:
Design a multi layer perceptron. Since this is a purely Feedforward network, you mustn't use any convolutional layers
Do not directly import or copy any existing models.
'''

# Given...
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x
        
# net = Net()


# input_size = the data set - training data size in pixels - 28 by 28
# hidden_size = 500
# num_classes = 10 according to hw instructions

input_size = 784
hidden_size = 500
num_classes = 10

# used the relu function to make my multi layer perceptron model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        #runs the function: linear on input and hidden
        self.func1 = nn.Linear(input_size, hidden_size)
        #runs the ReLU function
        self.relu = nn.ReLU()
        #runs the function: linear on hidden and numclasses
        self.func2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # flattens the input to prevent size mismatch errors, used geeksforgeeks to figure out the size error
        x = x.view(-1, 28 * 28)
        # calls the three functions used in the model to run the perceptron
        x = self.func1(x)
        x = self.relu(x)
        x = self.func2(x)
        return x

# runs model using the inputs: input_size, hidden_size, num_classes
net = Net(input_size, hidden_size, num_classes)

'''
PART 4:
Choose a good loss function and optimizer
'''
# Given...
# criterion = ''' Find a good loss function '''
# optimizer = ''' Choose a good optimizer and its hyperparamters '''

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

'''
PART 5:
Train your model!

'''
# Given...
# num_epochs = '''Choose the number of epochs '''
num_epochs = 10

training_losses = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    # for plotting
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # extracts the inputs and labels from the data set
        inputs, labels = data

        # zero the parameter gradients and performs the forward and backward steps
        optimizer.zero_grad()
        outputs = net(inputs)

        # Given...
        # loss = criterion(outputs, '''Fill in the blank''')
        loss = criterion(outputs, labels)

        # optimizes the function
        loss.backward()
        optimizer.step()

        # plotting variable: keeps track of the losses
        running_loss += loss.item()

        # plotting the training loss over time
        avg_loss = running_loss / len(trainloader)
        training_losses.append(avg_loss)

    print(f"Training loss: {running_loss}")

print('Finished Training')


'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        ''' Fill in this section '''
        #Creating a test set
        # 80% or higher

        #Extracts the images and labels from the data set provided
        images, labels = data
        outputs = net(images)
        # extracts the predicted values and 1 - loop variables used to help extract predicted values - towards data science
        _, predicted = torch.max(outputs.data, 1)
        # adds the labels and predicted labels to the total and correct values lists respectively
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: ', correct/total)


'''
PART 7:
Check the written portion. You need to generate some plots. 
'''


# plotting the training loss over time
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()



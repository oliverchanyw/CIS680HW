import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def run_nn(trainloader, testloader, learning_rate=0.5):

    # Define the net class
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
            self.fc1 = nn.Linear(128*2*2, 100)
            self.fc2 = nn.Linear(100, 10)

            # self.activation = nn.ReLU()
            self.activation = MySign.apply
            self.softmax = torch.nn.Softmax(dim=1)


        # N x C x H x W
        def forward(self, x):
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))
            x = self.activation(self.conv5(x))

            x = x.view(x.size(0), -1) # N x C
            x = self.activation(self.fc1(x))
            x = self.softmax(self.fc2(x))

            return x

    # Initialize the net
    net = Net()
    net.train()
    net = net.to(device)
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.00001)
    # create the Cross-entropy loss function
    criterion = nn.BCELoss()

    accuracies = []

    # Run the dataset through 4 times total
    for epoch in range(4):
        running_loss = 0.0

        # run the main training loop
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Create the ground truth labels vector 100 x 10
            groundtruth = torch.zeros(labels.size(0), 10)
            groundtruth = groundtruth.to(device)
            groundtruth[[i for i in range(labels.size(0))], labels] = 1

            # Set optimizer initial
            optimizer.zero_grad()

            # Use net
            net_out = net(inputs).to(device)

            # Calculate loss
            loss = criterion(net_out, groundtruth)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            predictions = net_out.argmax(dim=1)
            accuracy = torch.sum(torch.eq(predictions, labels)).item() / np.prod(labels.shape)
            accuracies.append(accuracy)

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.10f, accuracy: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, accuracy))
                running_loss = 0.0

    # Testing loop
    testaccuracies = []
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Use net
        net_out = net(inputs).to(device)

        # Calculate testing accuracy
        predictions = net_out.argmax(dim=1)
        accuracy = torch.sum(torch.eq(predictions, labels)).item() / np.prod(labels.shape)
        testaccuracies.append(accuracy)

    testaccuracy = sum(testaccuracies) / len(testaccuracies)
    # Return accuracy and loss
    return accuracies, testaccuracy

if __name__ == "__main__":
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].

    trainingtransform = transforms.Compose(
         [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])

    testingtransform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])

    trainset = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=trainingtransform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=testingtransform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ########################################################################
    # Run the neural net
    accuracies, testaccuracy = run_nn(trainloader, testloader)
    print(testaccuracy)
    # Plot accuracies
    plt.title("Classification accuracies (training) over minibatches")
    plt.plot(range(len(accuracies)), accuracies, 'r-')
    plt.show()

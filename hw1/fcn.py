import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def run_nn(imgs, labels, learning_rate=0.1, epochs=10000, log_interval=100, batchsize=64):

    # Define the net class
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(16, 4)
            self.fc2 = nn.Linear(4, 1)
            self.relu = nn.ReLU()
            self.sig = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sig(x)
            return x

    # Initialize the net
    net = Net()
    net.train()
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # create the L2 / Cross-entropy loss function
    criterion = nn.MSELoss()

    # Create a buffer to store the loss and accuracy
    losses = []
    accuracies = []

    # run the main training loop
    for epoch in range(epochs):
        # Reformat (batch size x 4 x 4) into (batch size x 16)
        imgs = imgs.view(-1, 16)
        optimizer.zero_grad()
        net_out = net(imgs)
        loss = criterion(net_out, labels)
        predictions = torch.where(net_out >= 0.5, torch.ones_like(labels), torch.zeros_like(labels))
        accuracy = torch.sum(torch.eq(predictions, labels)).item() / np.prod(labels.shape)

        loss.backward()
        optimizer.step()

        # print some information
        if (epoch % log_interval == 0):
            print("Epoch {}, loss = {}, accuracy = {}".format(epoch, loss.item(), accuracy))
            losses.append(loss.item())
            accuracies.append(accuracy)

        # Check if converged
        if (accuracy == 1.):
            print("Converged at epoch {} and loss of {}".format(epoch, loss))
            break

    # Return accuracy and loss
    return accuracies, losses

if __name__ == "__main__":
    # Load the data into torch
    imgs = np.load('datasets/random/random_imgs.npy')
    imgs = torch.from_numpy(imgs).float()
    labels = np.load('datasets/random/random_labs.npy')
    labels = torch.from_numpy(labels).float()
    labels = labels.view(64, 1)

    # Run the neural net
    accuracies, losses = run_nn(imgs, labels)

    # Plot
    plt.suptitle("L2 loss, ReLU activation")

    plt.subplot(211)
    plt.plot(range(0, len(accuracies)*100, 100), accuracies, 'ro')
    plt.title("Accuracies over epochs")

    plt.subplot(212)
    plt.plot(range(0, len(losses) * 100, 100), losses, 'bx')
    plt.title("Losses over epochs")

    plt.show()
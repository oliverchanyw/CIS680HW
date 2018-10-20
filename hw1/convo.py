import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def run_nn(imgs, labels, widths, learning_rate=0.1, epochs=10000, log_interval=1, batchsize=64):

    # Define the net class
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 7, stride=1, padding=3)
            self.conv2 = nn.Conv2d(16, 8, 7, stride=1, padding=3)
            self.fc1 = nn.Linear(256 * 8, 1)
            self.fc2 = nn.Linear(256 * 8, 1)
            self.relu = nn.ReLU(0.01)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x)).view(-1, 16 * 16 * 8)
            out2 = self.fc2(x)  # side output for L2 loss
            out1 = self.sig(self.fc1(x))  # standard output for BCE loss
            return out1, out2

    # Initialize the net
    net = Net()
    net.train()
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # create the L2 / Cross-entropy loss function
    entropy_criterion = nn.BCELoss()
    l2_criterion = nn.MSELoss()

    # Create a buffer to store the loss and accuracy
    losses1 = []
    losses2 = []
    accuracies1 = []
    accuracies2 = []

    # run the main training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        net_out1, net_out2 = net(imgs)
        loss1 = entropy_criterion(net_out1, labels)
        loss2 = l2_criterion(net_out2, widths)

        # Label accuracy
        predictions = torch.where(net_out1 >= 0.5, torch.ones_like(labels), torch.zeros_like(labels))
        accuracy1 = torch.sum(torch.eq(predictions, labels)).item() / np.prod(labels.shape)

        # Width accuracy
        accuracy2 = torch.sum(torch.where(abs(net_out2 - widths) < 0.5,
                                          torch.ones_like(widths), torch.zeros_like(widths))).item() \
                    / np.prod(widths.shape)

        # Backpropagate on both losses
        loss = loss1 + 0.01 * loss2
        loss.backward()
        optimizer.step()

        # print some information
        if (epoch % log_interval == 0):
            print("Epoch {}, loss = {}, accuracy = {}".format(epoch, loss.item(), accuracy2))
            accuracies1.append(accuracy1)
            accuracies2.append(accuracy2)
            losses1.append(loss1.item())
            losses2.append(loss2.item())

        # Check if converged (on width predictions)
        if (accuracy2 == 1.):
            print("Converged at epoch {} and loss of {}".format(epoch, loss))
            break

    # Return accuracy and loss
    return accuracies1, accuracies2, losses1, losses2

if __name__ == "__main__":
    # Load the data into torch
    imgs = np.load('datasets/detection/detection_imgs.npy')
    imgs = torch.from_numpy(imgs).float()
    imgs = imgs.view(64, 1, 16, 16)

    labels = np.load('datasets/detection/detection_labs.npy')
    labels = torch.from_numpy(labels).float()
    labels = labels.view(64, 1)

    widths = np.load('datasets/detection/detection_width.npy')
    widths = torch.from_numpy(widths).float()
    widths = widths.view(64, 1)

    # Run the neural net. 1 is label, 2 is width regression
    accuracies1, accuracies2, losses1, losses2 = run_nn(imgs, labels, widths)

    # Plot
    plt.suptitle("Detection dataset - losses and accuracy over training")

    plt.subplot(221)
    plt.plot(range(len(accuracies1)), accuracies1, 'r-')
    plt.title("Classification accuracies over epochs")

    plt.subplot(222)
    plt.plot(range(len(losses1)), losses1, 'b-')
    plt.title("Cross-entropy losses over epochs")

    plt.subplot(223)
    plt.plot(range(len(accuracies2)), accuracies2, 'r-')
    plt.title("Regression accuracies over epochs")

    plt.subplot(224)
    plt.plot(range(len(losses2)), losses2, 'b-')
    plt.title("L2 losses over epochs")

    plt.show()

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def run_nn(imgs, labels, learning_rate=0.1, epochs=10000, log_interval=500):

    # Define the net class
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 2)
            self.fc2 = nn.Linear(2, 1)
            self.tanh = nn.Tanh()
            self.sig = nn.Sigmoid()

        def forward(self, x):
            h = self.tanh(self.fc1(x))
            y = self.sig(self.fc2(h))
            return y, h

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
        optimizer.zero_grad()
        net_out, h = net(imgs)
        loss = criterion(net_out, labels)
        predictions = torch.where(net_out >= 0.5, torch.ones_like(labels), torch.zeros_like(labels))
        accuracy = torch.sum(torch.eq(predictions, labels)).item() / np.prod(labels.shape)

        loss.backward()
        optimizer.step()

        # print some information
        if (epoch % log_interval == 0):
            print("Epoch {}, loss = {}, accuracy = {}".format(epoch, loss.item(), accuracy))
            accuracies.append(accuracy)
            [w1, w2] = list(net.fc2.parameters())[0].detach().numpy().flatten().tolist()
            bias = list(net.fc2.parameters())[1].item()
            plotting_subroutine(h, w1, w2, bias, epoch)


        # Check if converged
        if (accuracy == 1.):
            print("Converged at epoch {} and loss of {}".format(epoch, loss))
            accuracies.append(accuracy)
            [w1, w2] = list(net.fc2.parameters())[0].detach().numpy().flatten().tolist()
            bias = list(net.fc2.parameters())[1].item()
            print(w1, w2)
            plotting_subroutine(h, w1, w2, bias, epoch)
            break

    # Return accuracy and loss
    return accuracies, losses

def plotting_subroutine(h, w1, w2, bias, epoch, log_interval=500):
    idx = math.ceil(epoch / log_interval)
    plt.figure(idx)
    plt.suptitle("Decision boundary in hidden layer at epoch {}".format(epoch))

    # Generate the linear separation equation
    x1 = np.linspace(-1.0, 1.0, num=100)
    x2 = -(bias + (w1 * x1)) / w2
    plt.plot(x2, x1, 'k-', linewidth=0.2)

    # Draw the initial points' maps
    plt.plot(h[0][0].item(), h[0][1].item(), 'r+')
    plt.plot(h[1][0].item(), h[1][1].item(), 'bo')
    plt.plot(h[2][0].item(), h[2][1].item(), 'r+')
    plt.plot(h[3][0].item(), h[3][1].item(), 'bo')

if __name__ == "__main__":
    # Load the data into torch
    imgs = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
    imgs = torch.from_numpy(imgs).float()
    imgs = imgs.view(4, 2)
    labels = np.array([1, 0, 1, 0])
    labels = torch.from_numpy(labels).float()
    labels = labels.view(4, 1)

    # Run the neural net
    accuracies, losses = run_nn(imgs, labels)

    # Plot only if we converged
    if (accuracies[-1] == 1.):
        plt.show()
    else:
        print("failed to converge")
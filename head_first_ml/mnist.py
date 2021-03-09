import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.optim as optim



data_train = datasets.MNIST(root = "./data/",
                            transform=ToTensor(),
                            train = True,
                            download = False)

data_test = datasets.MNIST(root="./data/",
                        transform = ToTensor(),
                        train = False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 1,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size = 1,
                                                shuffle = True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.W = nn.Parameter(torch.randn(784, 10))
        self.b = nn.Parameter(torch.zeros([10]))

    def forward(self, x):
        x = self.flatten(x)
        y = torch.matmul(x, self.W) + self.b
        # Applies the Softmax function to an n-dimensional 
        # input Tensor rescaling them so that the elements 
        # of the n-dimensional output Tensor lie in the range 
        # [0,1] and sum to 1.
        y = nn.Softmax(dim=1)(y)
        return y


model = NeuralNetwork()
# optimizer = optim.SGD(params=model.parameters(), lr=0.01)
optimizer = optim.Adam(params=model.parameters()) # Change grade desend function from SGD to Adam

def train(epoch):
    for batch_idx, (data, target) in enumerate(data_loader_train): 
        y = model(data)
        # y_real = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(target), value=1)
        # loss = torch.mean(-torch.sum(y_real * torch.log(y)))
        loss = nn.CrossEntropyLoss()(y.float(), target) # Use cross entrypy loss function to improve accuracy
        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                format(
                    epoch,
                    batch_idx * len(data),
                    len(data_loader_train.dataset),
                    100. * batch_idx / len(data_loader_train),
                    loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader_test): 
        y = model(data)
        # y_real = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(target), value=1)
        # test_loss += torch.mean(-torch.sum(y_real * torch.log(y)))
        test_loss += nn.CrossEntropyLoss()(y.float(), target)
        predictions = torch.argmax(y, dim = 1) # output tensor([5])
        correct += predictions == target # target tensor([1])
    accuracy = 100. * correct / len(data_loader_test.dataset)
    test_loss /= len(data_loader_test.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss.item(), accuracy.item() ))
        
                

if __name__ == '__main__':
    for epoch in range(1, 10):
        print("test num"+str(epoch))
        train(epoch)
        test()
    torch.save(model.state_dict(), "mnist_cnn.pt")
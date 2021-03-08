import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.W = nn.Parameter(torch.randn(784, 10))
        self.b = nn.Parameter(torch.zeros([10]))

    def forward(self, x):
        x = self.flatten(x)
        y = torch.matmul(x, self.W) + self.b
        y = nn.Softmax(dim=1)(y)
        return y


def main():
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


    model = NeuralNetwork()
    print(model)
    optimizer = optim.SGD([model.W, model.b], lr=0.01)
    # optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0)
    EPOCHS = 10
    history = {'Test loss':[],'Test Accuracy':[]}
    for epoch in range(1,EPOCHS + 1):
        model.train(True)
        for batch_idx, (data, target) in enumerate(data_loader_train): 
            y = model(data)
            y_real = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(target), value=1)
            loss = torch.mean(-torch.sum(y_real * torch.log(y)))
            predictions = torch.argmax(y, dim = 1)
            accuracy = torch.sum(predictions == target)/target.shape[0]
            # model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx%10000 == 0:
                correct,totalloss = 0,0
                model.train(False)
                for testData, testTarget in data_loader_test:
                    test_y = model(testData)
                    test_y_real = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(testTarget), value=1)
                    test_loss = torch.mean(-torch.sum(test_y_real * torch.log(test_y)))
                    predictions = torch.argmax(test_y,dim = 1)
                    totalloss += test_loss
                    correct += torch.sum(predictions == testTarget)
                testAccuracy = correct/(1 * len(data_loader_test))
                testloss = totalloss/len(data_loader_test)
                history['Test loss'].append(testloss.item())
                history['Test Accuracy'].append(testAccuracy.item())
                print("[%d/%d] loss: %.4f, Acc: %.4f, Test loss: %.4f, Test Acc: %.4f" % 
                                    (epoch,EPOCHS,loss.item(),accuracy.item(),testloss.item(),testAccuracy.item()))
                

if __name__ == '__main__':
    main()
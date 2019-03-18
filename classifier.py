import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
import math
from torchvision.models import inception_v3


n_epochs    = 10    # The number of times entire dataset is trained,iterations
batch_size_train = 8 # smaller number given better learning accuracy?? 64 - 91%  30-94% 16 - 96% 8-97%
batch_size_test = 1000   
learning_rate = 1e-3 # The speed of convergence，which is the default  the best for SGD based on https://medium.freecodecamp.org/how-to-pick-the-best-learning-rate-for-your-machine-learning-project-9c28865039a8
momentum=0.9 # 0.5 beofre with batch size train 8,given 96%  0.9-98%
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/sample.pt', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)



  # Old version CNN given 87% accuracy of FashinMnist,try to add layers and make it more clear
# new version CNN given 90% accuracy of FashionMnist.
class Net(nn.Module):
    def __init__(self):
         super(Net, self).__init__()
        
         self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(0.20),
         )
         self.fc = nn.Sequential(
            nn.Linear(32*8*8, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(p=0.5),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Softmax(dim=1)
         )
        
       # super(Net, self).__init__()
       # self.conv1 = nn.Conv2d(1, 15, kernel_size=5)
       # self.conv2 = nn.Conv2d(15, 20, kernel_size=5)
       # self.conv2_drop = nn.Dropout2d()
       # self.fc1 = nn.Linear(320, 50)
       # self.fc2 = nn.Linear(50, 10)

    def forward(self, x): 
      x = self.conv(x)
      print(x.size()) 
      x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
      return self.fc(x)
       
       # x = F.relu(F.max_pool2d(self.conv1(x), 2))
       # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
       # x = x.view(-1, 320)
       # x = F.relu(self.fc1(x))
       # x = F.dropout(x, training=self.training)
       # x = self.fc2(x)
       # return F.log_softmax(x)


model = Net()
print(model)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

use_cuda = True
if use_cuda and torch.cuda.is_available():
    network.cuda()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda and torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target) # we switch from negative log likelihood nll_loss https://pytorch.org/docs/stable/nn.html#cross-entropy  ,need to know exactly why
        loss.backward()
        optimizer.step()
    
    
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
     # torch.save(network.state_dict(), '/results/model.pth')
     # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
     for data, target in test_loader:
      if use_cuda and torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs + 1):
 # adjust_learning_rate(optimizer,epoch)
  train(n_epochs)
  test()

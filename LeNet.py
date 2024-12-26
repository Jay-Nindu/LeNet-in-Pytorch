import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class properC3(torch.nn.Module):
    def __init__(self):
        super(properC3, self).__init__()

        #16 feature maps need to be applied, each with unique inputs
        #list containing the number of inputs to each feature map
        self.size_of_map = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6]
        self.inputs_to_each = [[0,1,2],
                               [1,2,3],
                               [2,3,4],
                               [3,4,5],
                               [4,5,0],
                               [5,0,1],
                               [0,1,2,3],
                               [1,2,3,4],
                               [2,3,4,5],
                               [3,4,5,0],
                               [4,5,0,1],
                               [5,0,1,2],
                               [0,1,3,4],
                               [1,2,4,5],
                               [2,3,5,0],
                               [0,1,2,3,4,5]
                              ]

        #list containing each convolution, each having the appropriate number of input channels
        self.layers = torch.nn.ParameterList()
        for (i,size) in enumerate(self.size_of_map):
            self.layers.append(torch.nn.Conv2d(size, 1, (5,5), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))
            self.layers[i].weight = torch.nn.parameter.Parameter(data=torch.rand(self.layers[i].weight.size())*0.192-0.096, requires_grad=True)
            self.layers[i].bias = torch.nn.parameter.Parameter(data=torch.rand(self.layers[i].bias.size())*0.192-0.096, requires_grad=True)


    def forward(self, x):
        our_output = []
        for i, input_list in enumerate(self.inputs_to_each):
            a = [x[:,f,:,:] for f in input_list]
            currInput = torch.stack((a), dim=1)
            # print(currInput)
            our_output.append(self.layers[i](currInput))

        our_output = torch.cat(our_output, dim=1)
        return our_output


class subSample(torch.nn.Module):
    def __init__(self, inputs):
        super(subSample, self).__init__()
        self.inputs = inputs
        self.weight = torch.nn.ParameterList()
        self.bias = torch.nn.ParameterList()
        self.averagePool = torch.nn.AvgPool2d((2,2), stride=2)

        for i in range(self.inputs):
            self.weight.append(torch.nn.parameter.Parameter(data=torch.rand(1)*1.2 - 0.6, requires_grad=True))
            self.bias.append(torch.nn.parameter.Parameter(data = torch.tensor(0.0), requires_grad=True))

    def forward(self, x):
        our_output = []
        for i in range(self.inputs):
            currInput = x[:,i,:,:]
            # print(currInput)
            our_output.append(self.averagePool(currInput) * 4* self.weight[i] + self.bias[i])

        our_output = torch.stack(our_output, dim=1)
        # print(our_output)
        return our_output


class RBFUnit(torch.nn.Module):
    def __init__(self):
        super(RBFUnit, self).__init__()

        self.param = [[
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, 1, -1, -1],
        [-1, 1, 1, -1, 1, 1, -1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [-1, 1, 1, -1, 1, 1, -1],
        [-1, -1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],

        [
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, 1, 1, 1, -1, -1],
        [-1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],
        [
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, 1, 1, 1, 1, 1, -1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, 1, -1, -1],
    	[-1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],

        [
        [1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]

        ],
        [
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, 1, 1, -1, -1, 1, 1],
        [-1, 1, 1, -1, -1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, 1, 1, 1],
        [-1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1]
        ],

        [
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, -1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1],
        [-1, 1, 1, 1, 1, -1, -1],
        [-1, -1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],

        [
        [-1, -1, 1, 1, 1, 1, -1],
        [-1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1, -1],
        [1, 1, 1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1],
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],

        [
        [1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, -1, -1],
        [-1, -1, 1, 1, -1, -1, -1],
        [-1, -1, 1, 1, -1, -1, -1],
        [-1, -1, 1, 1, -1, -1, -1],
        [-1, -1, 1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],

        [
        [-1, 1, 1, 1, 1, 1, -1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [-1, 1, 1, 1, 1, 1, -1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [-1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ],

        [
        [-1, 1, 1, 1, 1, 1, -1],
        [1, 1, -1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, -1, 1, 1, 1],
        [1, 1, -1, -1, 1, 1, 1],
        [-1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, -1],
        [-1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
        ]
       ]

        self.param = torch.Tensor(self.param)
        self.param = torch.flatten(self.param, start_dim=1)
        # print(self.param)
        self.weight = torch.nn.ParameterList()
        for i in range(10):
            # self.weight.append(torch.nn.parameter.Parameter(data=torch.rand(84)*0.05714 - 0.02857, requires_grad=True))
            self.weight.append(torch.nn.parameter.Parameter(data=self.param[i], requires_grad=True))


    def forward(self, x):
        my_output = []
        for slice in range(x.size()[0]):
            my_temp_output = []
            for i in range(10):
                delta = x[slice,:]-self.weight[i]
                my_temp_output.append(torch.dot(delta, delta))
            my_output.append(torch.stack(my_temp_output))
        return torch.stack(my_output)


class leNet(torch.nn.Module):
    def __init__(self):
        super(leNet, self).__init__()

        self.activation = torch.nn.Tanh() #F(a) = A * tanh(S * a). A = 1.7159. S=2/3

        self.C1 = torch.nn.Conv2d(1, 6, (5,5), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.C1.weight = torch.nn.parameter.Parameter(data=torch.rand(self.C1.weight.size())*0.192-0.096, requires_grad=True)
        self.C1.bias = torch.nn.parameter.Parameter(data=torch.rand(self.C1.bias.size())*0.192-0.096, requires_grad=True)

        self.S2 = subSample(6)
        self.C3 = properC3()
        self.S4 = subSample(16)
        self.C5 = torch.nn.Conv2d(16, 120, (5,5), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.C5.weight = torch.nn.parameter.Parameter(data=torch.rand(self.C5.weight.size())*0.192-0.096, requires_grad=True)
        self.C5.bias = torch.nn.parameter.Parameter(data=torch.rand(self.C5.bias.size())*0.192-0.096, requires_grad=True)

        self.F6 = torch.nn.Linear(120, 84, bias=True, device=None, dtype=None)
        self.F6.weight = torch.nn.parameter.Parameter(data=torch.rand(self.F6.weight.size())*0.192-0.096, requires_grad=True)
        self.F6.bias = torch.nn.parameter.Parameter(data=torch.rand(self.F6.bias.size())*0.192-0.096, requires_grad=True)


        # setting the weights for convolutional layers C1 and C5, and linear layer F6
        self.RBF = RBFUnit()

    def forward(self, x):
        #Feature extraction:
        ## first convolution -> subsampling layer
        x = self.C1(x)
        x = self.S2(x)
        x =  1.7159 * self.activation(2/3*x)

        ## second convultion -> subsampling layer
        x = self.C3(x)
        x = self.S4(x)
        x = 1.7159 * self.activation(2/3*x)

        x = self.C5(x)
        #After C5. flatten to create inputs to fully connected layer (linear) -> F6
        x = torch.flatten(x, start_dim=1)

        x = self.F6(x)
        x = self.RBF(x)
        #After F6 need to apply gaussian garbage
        return x


class RBFCost(torch.nn.Module):
  def __init__(self):
    super(RBFCost, self).__init__()
    self.register_buffer('j', torch.ones(1))


  def forward(self, x, y):

     ## x being pred, y being actual

    correctPred = x[torch.arange(x.size(0)),y]
    incorrectError = torch.log(torch.exp(-self.j) + torch.sum(torch.exp(-1*x), dim=1))

    totalError = correctPred + incorrectError
    # _, accuracy =torch.max(x, dim=1)
    # accuracy = accuracy[torch.eq(accuracy, y)].size(0)
    return totalError

#data handling

root = os.getcwd()
splits = ["train", "test"]

myTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2, fill=0)]
)


MNISTdata = {x : datasets.MNIST(root = root, train = (x=="train"), download=(not os.path.exists("MNIST")), transform=myTransforms) for x in splits}
MNISTloader = {x : DataLoader(MNISTdata[x], batch_size=32) for x in splits}


myNet = leNet().to(device)
myError = RBFCost()
myError = myError.to(device)
optimizer = torch.optim.SGD(myNet.parameters(), lr=0.005)

length = {x: len(MNISTdata[x]) for x in splits}
hist = []

for epoch in range(20):
  print(f"Epoch {epoch}: --------------------")
  for mode in splits:
    if mode == 'train':
      myNet.train()  # Set model to training mode
    else:
      myNet.eval()   # Set model to evaluate mode
    currLoss = 0
    optimizer.zero_grad()

    for x,y in MNISTloader[mode]:
      x = x.to(device)
      y = y.to(device)
      pred = myNet(x)

      loss = myError(pred, y)
      loss = loss.sum()
      loss = (1/length[mode]) * loss
      currLoss += loss.item()

      if(mode == "train"):
        loss.backward()

    print(f"{mode} error: {currLoss}")
    if(mode == "train"):
      optimizer.step()
    else:
      hist.append(currLoss)

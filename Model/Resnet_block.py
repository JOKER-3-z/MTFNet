import torch
from torch import nn

import math
def conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            residual=self.downsample(x)
        out=self.relu(out+residual)
        return out

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        self.conv1=nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size=1,bias=False),
            nn.Batchnorm2d(planes)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.Batchnorm2d(planes)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(inplanes,planes*Bottleneck.expansion,kernel_size=1,bias=False),
            nn.Batchnorm2d(planes*Bottleneck.expansion)
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride
    def forward(self,x):
        residual=x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(x))
        out=self.conv3(out)

        if self.downsample is not None:
            residual=self.downsample(x)
        return self.relu(out+residual)
    
class ResNet(nn.Module):
    def __init__(self,block,layers):
        super(ResNet,self).__init__()
        self.inplanes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self.__make_layer(block,64,layers[0])
        self.layer2=self.__make_layer(block,128,layers[1],stride=2)
        self.layer3=self.__make_layer(block,256,layers[2],stride=2)
        self.layer4=self.__make_layer(block,512,layers[3],stride=2)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]* m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif  isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __make_layer(self,block,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes != planes*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                          kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out1=self.layer1(out)#64,56,56
        out2=self.layer2(out1)#128，28，28
        out3=self.layer3(out2)#256，14，14
        out4=self.layer4(out3) #512，7，7

        return out1,out2,out3,out4
def cal(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def build_resnet18(parameterDir):
    model=ResNet(BasicBlock,[2,2,2,2])
    if parameterDir is not None:
        model=LoadParameter(model,parameterDir)
    return model

def build_resnet34(parameterDir):
    model=ResNet(BasicBlock,[3,4,6,3])
    if parameterDir is not None:
        model=LoadParameter(model,parameterDir)
    return model

def LoadParameter(_structure, _parameterDir):
    #print(_structure)
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']#model_state_dict #state_dict
    model_state_dict = _structure.state_dict()
    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (key == 'module.feature.bias')):

            pass
        elif( 'classifier' in key): 
            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
    #print(pretrained_state_dict)
    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure,device_ids=[0])

    return model.to('cuda:0')

if __name__ == "__main__":
    pass
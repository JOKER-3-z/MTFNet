import math
import torch
import numpy as np
from torch import nn
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
from Model.Resnet_block import build_resnet18

from torchsummary import summary

class PosEncoding(torch.nn.Module):
    def __init__(self,max_seq_len,d_model=512,device="cpu"):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000,2.0 * (j//2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)]
        )
        pos_enc[:,0::2] = np.sin(pos_enc[:,0::2])
        pos_enc[:,1::2] = np.cos(pos_enc[:,1::2])
        pos_enc = pos_enc.astype(np.float32)
        self.pos_enc = torch.nn.Embedding(max_seq_len,d_model)
        self.pos_enc.weight = torch.nn.Parameter(torch.from_numpy(pos_enc),requires_grad=False)
        self.device=device

    def forward(self, input_len):
        '''

        :param input_len: [7,7,7,7,...,7] shape=[batch_size]
        :return:
        '''
        input_pos = torch.tensor([list(range(0,len)) for len in input_len]).to(self.device)
        return self.pos_enc(input_pos)

class GSConv(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,g=1,act=True):
        super().__init__()
        c_=c2//2
        self.conv1=nn.Conv2d(c1,c_,k,s,groups=1, bias=act)
        self.conv2=nn.Conv2d(c_,c_,k,1,groups=c_, bias=act)
    def forward(self,x):
        x1=self.conv1(x)
        x2=torch.cat((x1,self.conv2(x1)),1)
        #shuffle
        b,n,h,w=x2.data.size()
        b_n=b*n//2
        y=x2.reshape(b_n,2,h*w)
        y=y.permute(1,0,2)
        y=y.reshape(2,-1,n//2,h,w)
        return torch.cat((y[0],y[1]),1)
    
class GSConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,s1=1,s2=1):
        super().__init__()
        self.gsc1=GSConv(in_channel,out_channel,s=s1)
        self.gsc2=GSConv(out_channel,out_channel,s=s2)
    def forward(self,x):
        return self.gsc2(self.gsc1(x))

class FUnit(nn.Module):
    def __init__(self,in_ch1,in_ch2,in_ch3,out_ch,s=[2,2,2,1,1,1]):
        super().__init__()
        self.GSBFF=nn.Sequential(*[GSConvBlock(in_ch1,in_ch3,s1=s[0],s2=s[1]),
                                  nn.BatchNorm2d(in_ch3),nn.ReLU(inplace=True)])
        self.GSBFD=nn.Sequential(*[GSConvBlock(in_ch2,in_ch3,s1=s[2],s2=s[3]),
                                    nn.BatchNorm2d(in_ch3),nn.ReLU(inplace=True)])
        self.GSBCE=nn.Sequential(*[GSConvBlock(in_ch3,in_ch3,s1=s[4],s2=s[5]),
                            nn.BatchNorm2d(in_ch3),nn.ReLU(inplace=True)])
        self.GSBFC=nn.Sequential(*[GSConvBlock(in_ch3,out_ch),
                                  nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True)])
    def forward(self,x1,x2,x3):
        y1=self.GSBFF(x1)
        y2=self.GSBFD(x2)
        y3=self.GSBCE(x3)
        return self.GSBFC(y1+y2+y3)
    
class Multi_Scale(nn.Module):
    def __init__(self,parameterDir=None):
        super().__init__()
        self.feature=build_resnet18(parameterDir)
        self.FU1=FUnit(64,128,256,256) #256,14,14
        self.FU2=FUnit(128,256,512,512) #512,7,7
        self.FU3=FUnit(256,256,512,512,s=[2,1,2,1,1,1]) # 1024 7 7
        #self.avg=nn.AdaptiveAvgPool2d(1)
        self.maxp=nn.MaxPool2d(kernel_size=7)
    def forward(self,x):
        f1,f2,f3,f4=self.feature(x)
        #return self.avg(f4)
        g1=self.FU1(f1,f2,f3)
        g2=self.FU2(f2,f3,f4)
        g3=self.FU3(f3,g1,f4)
        return self.maxp(g3)

class Permutation(nn.Module):
    def __init__(self,classes,num_image):
        super(Permutation, self).__init__()
        self.classes = classes
        self.num_image=num_image

        self.classifier = nn.Sequential(*[nn.Linear( self.num_image*512,2048),nn.BatchNorm1d(2048),nn.ReLU(inplace=True),
                                          nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True),
                                          nn.Linear(512,self.classes)])


        self.weights_init(self.classifier)

    def forward(self, input):
        output = self.classifier(input)
        return output

    def weights_init(self,model):
        if type(model) in [nn.ConvTranspose2d, nn.Linear]:
            nn.init.xavier_normal(model.weight.data)
            nn.init.constant(model.bias.data, 0.1)

class MTFsNet(nn.Module):
    def __init__(self,num_frame=7,num_class=7,d_model=512,nhead=4,depth=2,dropout=0.2,device="cpu",parameterDir=None,draw_weight=True):
        super(MTFsNet,self).__init__()
        self.d_model=d_model
        self.num_frame=num_frame
        self.num_class=num_class
        self.device=device
        #self.per_classes=
        self.draw_weight=draw_weight
        self.feature=Multi_Scale(parameterDir)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.pos_enc = PosEncoding(max_seq_len=self.num_frame, d_model=self.d_model,device=self.device)
        self.classifier=nn.Sequential(*[nn.Linear(512, 128), nn.ReLU(inplace=True),
                                          nn.Dropout(0.4),
                                          nn.Sequential(nn.Linear(128, self.num_class),nn.Softmax(dim=1))])
        #self.per_branch=Permutation(classes=self.num_class,num_image=num_frame)
        self.avgpool=nn.AvgPool1d(kernel_size=7)
        #self.maxp=nn.MaxPool2d(kernel_size=7)


    def forward(self,x,per=[0,1,2,3,4,5,6]):
        x = x.contiguous().view(-1, 3, 224, 224)
        x=self.feature(x)#[bs,nf,512,1,1]
        #print(x.shape)#[bs,512,1,1]
        x = x.squeeze(3).squeeze(2)
        x = rearrange(x,'(b f) c-> b f c',f=self.num_frame)
        ori_video_feature = x
        b = int(ori_video_feature.size(0))
        pos = torch.tensor([self.num_frame] * b)
        pos_enc = self.pos_enc(pos)
        for i in range(b):
            pos_enc[i]=pos_enc[i][per[i]]

        x = ori_video_feature + pos_enc
        x = x.permute(1, 0, 2)
        #print(x.shape)
        emotion_clip_feature= self.transformer(x) #.permute(1, 2, 0)
        #print(emotion_clip_feature.shape)
        emotion_clip_feature = emotion_clip_feature.permute(1,2,0)
        #print(emotion_clip_feature.shape)
        emotion_clip_feature=self.avgpool(emotion_clip_feature).squeeze(2)
        #print(emotion_clip_feature.shape)
        output = self.classifier(emotion_clip_feature)
        
        #ori_video_feature_detach=ori_video_feature.detach()
        #per_feature = ori_video_feature_detach +emotion_clip_feature.unsqueeze(1)
        #per_out=self.per_branch(per_feature.view(b,-1))
        return output

class FERModel():
    def __init__(self,labels,num_class=7,data_transforms=None,device_id='cpu'):
        self.device = torch.device(device_id)
        self.data_transforms = data_transforms                        
        self.labels = labels
        self.model = MTFsNet(num_class=num_class,device=device_id)
    
        self.log=nn.Softmax(dim=1)#用来算概率
        self.model.to(self.device)
        self.model.eval()

    def detect(self,src):
        if src is not None:
            fer=self.fer(src)
        return fer
          
    def fer(self, imgs):
        input=torch.stack(imgs,dim=1).to(self.device)
        with torch.set_grad_enabled(False):
            out= self.model(input)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]
            return label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    #Mul_scale = 12331328（FU1:226560 , FU2:928256,resnet18:11176512）
    
    model=MTFsNet().to("cpu")
    #x=torch.randn([2,7,3,224,224]).to("cpu")
    #t,a=model(x)
    print(model)
    #print(count_parameters(model))

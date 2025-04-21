import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torchvision import transforms,datasets
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


import os
import math
import random
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import load_materials
from sklearn.metrics import confusion_matrix

from models.resnet_encoder import GLNet

from torchsummary import summary
from torch.backends import cudnn
parser=argparse.ArgumentParser()
parser.add_argument("--datapath",default='../Datasets_dfer/DFEW',type=str,help='path of dataset')
parser.add_argument("--resume",default=None,type=str,help='loading pretrained model')
parser.add_argument("--eval",default=False,type=bool,help='use the eval first')
parser.add_argument('--Isvisible',type=bool,default=False)
parser.add_argument("--model_name",default="out_data",type=str,help='model_name to save')
parser.add_argument('--snippets', type=int, default=7, help='the number of snippets')
parser.add_argument('--per_snippets', type=int, default=5, help='the number of per snippets')
parser.add_argument('--use_norm', action='store_false')
parser.add_argument('--d_model',type=int,default=512)
parser.add_argument('--nhead',type=int,default=4)
parser.add_argument('--encoder_nums',type=int,default=3)
parser.add_argument('--gpu_ids',type=str,default='0',help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--num_class",default=7,type=int,help='how many classes need to output')
parser.add_argument("--lr",default=0.0001,type=float,help='learning rate')
parser.add_argument("--bs",default=48,type=int,help='batch_size')
parser.add_argument("--epochs",default=100,type=int,help='epochs')
parser.add_argument("--step_lr",default=0.001,type=float,help='LR decay')
parser.add_argument("--step_size",default=5,type=float,help='how many epochs to  lr_decay')
parser.add_argument('--parameterDir',type=str,default="./save_point/seed_3407_67.727_resnet18-encorder-fsop.pth")
#parser.add_argument('--parameterDir',type=str,default='./save_point/epochs:75_82.497_FFN.pth')
args=parser.parse_args()
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
def set_seed(seed: int = 1) -> None:
    """
    设置相关函数的随机数种子
    :param seed: 随机数种子
    :return: None
    """

    # 随机数种子设定
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    torch.backends.cudnn.deterministic = True

    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    torch.backends.cudnn.benchmark = False
    
def calculate_metrics(predictions_list, true_labels_list):
    all_predictions = np.concatenate(predictions_list, axis=0)
    all_true_labels = np.concatenate(true_labels_list, axis=0)
    #print(all_predictions.size)
    #print(all_true_labels.size)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    #混淆矩阵归一化
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_normalized = np.round(conf_matrix_normalized, 4)
    # 计算WAR
    war = weighted_accuracy_rate(conf_matrix)
    
    # 计算UAR
    uar = unweighted_accuracy_rate(conf_matrix)
    data_ditc={'confusion_matrix':conf_matrix_normalized.tolist() , 'War':war ,'Uar':uar }
    return data_ditc

def weighted_accuracy_rate(conf_matrix):
    total_samples = np.sum(conf_matrix)
    diagonal_sum = np.sum(np.diag(conf_matrix))
    return diagonal_sum / total_samples

def unweighted_accuracy_rate(conf_matrix):
    class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return np.mean(class_accuracy)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def getshufflelist(per_label):
    shufflelist=[
        [0,4, 3, 2, 6, 1, 5],
        [5, 3, 1, 6 ,4 ,0, 2],
        [0 ,4 ,3 ,5 ,2 ,6 ,1],
        [5 ,4, 3, 2, 1, 0, 6],
        [5 ,1, 2, 4, 0 ,6, 3],
        [0 ,5 ,2 ,6 ,4 ,1 ,3],
        [3 ,1 ,5 ,0 ,6 ,2 ,4],
        [2 ,3 ,0 ,4 ,1 ,6 ,5],
        [5 ,0 ,4 ,1 ,3 ,2 ,6],
        [0 ,3 ,2 ,6 ,4 ,5 ,1]
    ]
    per_list=shufflelist[per_label]
    return per_list

def getsort(root):
    li=os.listdir(root)
    li.sort(key=lambda x : int(x.split('.')[0].split('_')[1]))
    return li

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits

class MMEW(data.Dataset):
    def __init__(self, root, phase, transform = None):
        label_dict={'anger':0, 'fear':1,'disgust': 2,'happiness': 3, 'sadness':4,'surprise' :5}
        file_paths=[]
        labels=[]
        assert phase=='train' or phase=='val' or phase=='test'
        emolabel=os.listdir(os.path.join(root,phase))
        for emo in emolabel:
            rt=os.path.join(root,phase,emo)
            im_list=os.listdir(rt)
            for ifile in im_list:
                file_paths.append(os.path.join(rt,ifile))
            labels.extend([label_dict[emo]]*len(im_list))
        self.file_paths=file_paths
        self.labels=labels
        self.transform=transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        imlist=os.listdir(path)
        imlist.sort(key=lambda x:int(x.split('.')[0].split('-')[2]))
        img=[]
        for im in imlist:
            #print(os.path.join(path,im))
            image = Image.open(os.path.join(path,im)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            img.append(image)
        return img,label

class Label_data(data.Dataset):
    def __init__(self,root,labelf,data_type=None,transform = None):
        label_dict={'EMO_confuse':0,'EMO_bored':1,'EMO_interest':2,'EMO_happy':3}
        file_paths=[]
        labels=[]
        df=pd.read_csv(labelf)
        datafile=df['file'].values
        self.transform=transform
        data_file=[]
        for i in datafile:
            data_file.append(os.path.join(root,'data',str(i)))
        self.data_file=data_file
        self.labels=df['label'].values
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.data_file[idx]
        label = self.labels[idx]
        imlist=os.listdir(path)
        #imlist.sort(key=lambda x:int(x.split('.')[0].split('-')[2]))
        img=[]
        shuffle_list=[0,1,2,3,4,5,6]
        for im in imlist:
            #print(os.path.join(path,im))
            image = Image.open(os.path.join(path,im)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            img.append(image)
        return {'img':img,'class_label':label,'per_list':shuffle_list,'per_label':1}

class CKPLUS(data.Dataset):
    def __init__():
        pass
    def __len__():
        pass
    def __getitem__():
        pass

class DFEW(data.Dataset):
    def __init__(self,root,labelf,data_type=None,transform = None):
        #root 软链接 zy/DataSets/DFEW -> ./datasets/DFEW 
        label_dict={'Happy':0,'Sad': 1,'Neutral': 2, 'Angry':3,'Surprise' :4
                    , 'Disgust':5,'Fear' :6}
        self.root=root
        self.data_type=data_type
        self.transform=transform
        df=pd.read_csv(labelf)
        self.datafile=df['video_name'].values
        self.labels=df['label'].values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        imfile=str(self.datafile[idx])
        imfile='0'*(5-len(imfile))+imfile
        label= self.labels[idx]-1
        imgpath=os.path.join(self.root,imfile)
        imglist=os.listdir(imgpath)
        imglist.sort(key=lambda x : int(x.split('.')[0]))
        img=[]
        per_label=random.randint(0,6)
        label_rep={0:2,1:4,2:6,3:8,4:10,5:12,6:14}
        if self.data_type=='train':
            shuffle_list=getshufflelist(per_label)
        else :
            shuffle_list=[0,1,2,3,4,5,6]
        for i in shuffle_list:
            image = Image.open(os.path.join(imgpath,imglist[label_rep[i]])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            img.append(image)
        return  {'img':img,'class_label':label,'per_list':shuffle_list,'per_label':per_label}
        return img,label

def train_one_epoch(model,train_loader,losses,optimizer):
    model.train()
    loss_recorder_cl=AverageMeter()
    loss_recorder_per=AverageMeter()
    acc_recorder_cla=AverageMeter()
    for i,(batch) in enumerate(train_loader):
        optimizer.zero_grad()
        input=torch.stack(batch['img'],dim=1)
        per=torch.stack(batch['per_list'],dim=1)
        #print(input.shape)
        target=batch['class_label'].to(DEVICE)
        per_label=batch['per_label'].to(DEVICE)

        B, N,C, H, W = input.shape
        class_out,per_out=model(input,per)
        #print(per_label.shape)
        loss_cla = losses(class_out,target)
        #loss_per=losses(per_out,per_label)
        loss=loss_cla
        #print(loss.shape)
        #loss_recorder_per.update(loss_per.item(),n=input.shape[0])
        loss_recorder_cl.update(loss_cla.item(),n=input.shape[0])
        acc_cla = accuracy(class_out, target)

        acc_recorder_cla.update(acc_cla[0],n=input.shape[0])
        loss.backward()
        optimizer.step()
    #pa,ca=loss_recorder_per.avg,loss_recorder_cl.avg
    ca=loss_recorder_cl.avg
    print(" loss_cla:" +str(ca))
    loss=ca
    acc_cla=acc_recorder_cla.avg

    return loss,acc_cla

def eval(model,val_loader,losses):
    model.eval()
    loss_recoder=AverageMeter()
    acc_recorder_cla=AverageMeter()
    
    predictions_list = []  # 存储每个batch的预测结果
    true_labels_list = []  # 存储每个batch的真实标签
    with torch.no_grad():
        for i,(batch) in enumerate(val_loader):
            #print(len(images))
            input=torch.stack(batch['img'],dim=1).to(DEVICE)
            per=torch.stack(batch['per_list'],dim=1)
            target=batch['class_label'].to(DEVICE)
            per_label=batch['per_label'].to(DEVICE)
            B, N,C, H, W = input.shape
            class_out,_=model(input,per)
            
            predictions_list.append(class_out.argmax(axis=1).detach().cpu().numpy()) 
            true_labels_list.append(target.cpu().numpy()) 
            #print('-----------------out-------------')
            _, pred = class_out.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            loss_cla = losses(class_out,target)
            loss=loss_cla

            #print(loss.shape)
            loss_recoder.update(loss.item(),n=input.shape[0])
            acc_cla = accuracy(class_out, target)

            acc_recorder_cla.update(acc_cla[0],n=input.shape[0])

        loss=loss_recoder.avg
        acc_cla=acc_recorder_cla.avg
        conf_matrix_war_uar_dict = calculate_metrics(predictions_list, true_labels_list)
    return conf_matrix_war_uar_dict,loss,acc_cla

def test(model,loader,epoch):
    '''
    draw feature extreact in model
    '''
    targets=[]
    preds=[]
    model.eval()
    with torch.no_grad():
        for i,(images,target) in enumerate(loader):
            input=torch.stack(images,dim=1)
            target=target.to(DEVICE)
            out=model(input)
            #print('-----------------out-------------')
            _, pred = out.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            targets.extend(target.tolist())
            preds.extend(pred.tolist())
        con={'epoch: '+str(epoch):{'target':targets,'preds':preds}}

        with open('result.json','a') as f:
            context=json.dumps(con)
            f.write(context)


def main(idx,labelfT,labelfV,forzen_method=None):
    #超参数定义
    set_seed(3407)
    batch_size=args.bs
    lr=args.lr
    epochs=args.epochs
    datapath=args.datapath
    pretrained_model=args.resume
    Iseval=args.eval
    Isvisible=args.Isvisible
    #dataloader
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])  
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    train_set = Label_data('./label_data',labelf=labelfT,data_type='train', transform = data_transforms_train)
    val_set = Label_data('./label_data', labelf=labelfV ,transform = data_transforms_val)
    

    print('length of train_set : '+str(train_set.__len__()))
    print('length of val_set : '+str(val_set.__len__()))

    train_loader=data.DataLoader(train_set,batch_size=batch_size,num_workers=4,shuffle=True, pin_memory=True,drop_last=True)
    val_loader=data.DataLoader(val_set,batch_size=batch_size,num_workers=4,shuffle=False, pin_memory=True)

    #model define
    #model=MMEW_EST.resnet18_EST(classfier=7,img_num=7)
    #model=fan.resnet18_at(pretrained=True,num_pair=16)
    #model=FF_Trans.resnet18_at(pretrained=True,num_pair=16)
    #model=STFormer.GenerateModel()
    #model=IAL.GCA_IAL()
    model=GLNet(num_class=4)
    model.to(DEVICE)
    model = load_materials.LoadParameter(model, args.parameterDir)
    for param in model.named_parameters():
        if 'classifier' in param[0] and  'branch' not in  param[0]:
            pass
        else:
            param[1].requires_grad = False

    #optimizer,loss
    class_weights = torch.tensor([4, 2, 0.8, 1.0]).to("cuda:0")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    per_branch_params = list(map(id, model.per_branch.parameters()))
    base_params = filter(lambda p: id(p) not in per_branch_params and p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.per_branch.parameters(), 'lr': lr,"weight_decay":0.05}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)
    warmup_scheduler.last_step = -1
    

    if Iseval:
        val_loss,val_acc=eval(model,val_loader,criterion)
        tqdm.write('[initial weights] eval accuracy: %.4f. Loss: %.3f. LR %.6f' % (val_acc, val_loss,optimizer.param_groups[0]['lr']))
    
    best_acc=-1
    best_epoch=-1
    model_name=args.model_name
    if not os.path.exists("./save_point"):
        os.mkdir("./save_point")
    
    Acc={'val':[],'train':[]}
    Loss={'val':[],'train':[]}
    num_file=len(os.listdir('./logs'))+1
    summaryWriter = SummaryWriter("logs/log"+str(num_file))
    for epoch in range(1,epochs+1):
        train_loss, train_acc = train_one_epoch(model,train_loader,criterion,optimizer)
        summaryWriter.add_scalar(f'floder{idx}_traion_loss', train_loss, epoch)
        summaryWriter.add_scalar(f'floder{idx}_train_acc', train_acc, epoch)
       
        
        Acc['train'].append(train_acc)
        Loss['train'].append(train_loss)
        tqdm.write('[Epoch %d] Training accuracy: %.4f Loss: %.3f. LR %.10f' % (epoch, train_acc, train_loss,optimizer.param_groups[0]['lr']))
        matrix_dict,val_loss,val_acc=eval(model,val_loader,criterion)
        summaryWriter.add_scalar(f'floder{idx}_val_loss', val_loss, epoch)
        summaryWriter.add_scalar(f'floder{idx}_val_acc', val_acc, epoch)
        Acc['val'].append(val_acc.cpu().numpy())
        Loss['val'].append(val_loss)
        tqdm.write('[Epoch %d] eval accuracy: %.4f Loss: %.3f. LR %.10f' % (epoch, val_acc, val_loss,optimizer.param_groups[0]['lr']))
        if val_acc > best_acc:
            best_epoch=epoch
            best_acc=val_acc
            #保存权重
            pthname='epochs:{}_{:.3f}_{}.pth'.format(epoch+1,best_acc.item(),model_name)
            torch.save({'iter': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        os.path.join('./save_point', pthname))
            #test(model,val_loader,epoch)
            #保存混淆矩阵
            with open(os.path.join("./matrixdata",pthname+'.json'), 'w') as f:
                json.dump(matrix_dict, f)
        tqdm.write('Best acc : %.3f , best epoch : %d'%(best_acc,best_epoch))
        scheduler.step()   
    if Isvisible==True:
        x=[i for i in range(0,len(Loss['val']))]
        plt.subplot(2, 1, 1)

        plt.plot(x, Acc['val'] , '.-',label='val')
        plt.plot(x, Acc['train'] , 'o-',label='train')
        plt.title('accuracy vs. epoches')
        plt.ylabel('accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x,Loss['val'] , '.-',label='val')
        plt.plot(x,Loss['train'] , 'o-',label='train')
        plt.xlabel('loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        plt.savefig("jpefile/accuracy_loss"+str(idx)+".jpg")
    return best_acc


def run5_floder(root):
    Tlabelfolder=os.path.join(root,'trainset')
    Vlabelfolder=os.path.join(root,'testset')
    Tlabel=os.listdir(Tlabelfolder)
    Vlabel=os.listdir(Vlabelfolder)
    Tlabel.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    Vlabel.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    Acc=[]
    with open('log.txt','a') as f:
            f.write(args.model_name +'\n')
    for i in range(len(Tlabel)):
        labelt=os.path.join(Tlabelfolder,Tlabel[i])
        labelv=os.path.join(Vlabelfolder,Vlabel[i])
        acc=main(i,labelt,labelv)
        Acc.append(acc)
        with open('log.txt','a') as f:
            f.write('set_'+str(i)+':       '+str(acc)+'\n')
    with open('log.txt','a') as f:
        f.write('Average of 5 folder:  '+str(sum(Acc)/len(Acc))+'\n')
    print(sum(Acc)/len(Acc))
if __name__ == "__main__":
    #run5_floder('../Datasets_dfer/labelFile')
    
    acc=main(0,'./label_data/label.csv','./label_data/label.csv')
    with open('log.txt','a') as f:
        f.write(args.model_name+'\n')
        f.write('set_'+str(1)+':       '+str(acc)+'\n')

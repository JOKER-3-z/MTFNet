from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
from data.random_shuffle_dataset import RandomShuffleDataset
import utils

import os
import PIL.Image as Image

class MMEW(torch.utils.data.Dataset):
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
        img=[]
        for im in imlist:
            image = Image.open(os.path.join(path,im)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            img.append(image)
        #print(label)
        return img,label
def getMMEW(opt):
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),])  
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    train_set = MMEW(opt.datapath, phase = 'train', transform = data_transforms_train)    
    val_set = MMEW(opt.datapath, phase = 'val', transform = data_transforms_val)   
    


    train_loader=torch.utils.data.DataLoader(train_set,batch_size=64,num_workers=4,shuffle=True, pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_set,batch_size=64,num_workers=4,shuffle=False, pin_memory=True)
    return train_loader,val_loader

def LoadDataset(opt):
    cate2label = utils.cate2label(opt.dataset_name)

    train_dataset = RandomShuffleDataset(
        video_root=opt.train_video_root,
        video_list=opt.train_list_root,
        rectify_label=cate2label,
        isTrain= True,
        transform=transforms.Compose([transforms.ToTensor()]),
        opt=opt
    )

    val_dataset = RandomShuffleDataset(
        video_root=opt.test_video_root,
        video_list=opt.test_list_root,
        rectify_label=cate2label,
        isTrain = False,
        transform=transforms.Compose([transforms.ToTensor()]),
        opt=opt
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_threads,
         pin_memory=True, drop_last=Flase)   #True若数据集大小不能被batch_size整除，则删除最后一个不完整的批处理。

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size, shuffle=False,num_workers=opt.num_threads,
         pin_memory=True)

    return train_loader, val_loader


def LoadParameter(_structure, _parameterDir):
    #print(_structure)
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']#model_state_dict #state_dict
    model_state_dict = _structure.state_dict()
    for key in pretrained_state_dict:
        if 'classifier' in key and 'branch' not in key:
            pass
        elif 'feature.feature' in key:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
        else:
            model_state_dict[key] = pretrained_state_dict[key]
    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure,device_ids=[0])
    return model.to('cuda:0')

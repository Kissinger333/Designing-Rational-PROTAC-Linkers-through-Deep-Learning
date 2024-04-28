from model import *
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from sklearn.model_selection import train_test_split

class ProDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.transform = transform
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        key = self.keys[index]
        feature2D_tensor = key
        label = self.data_dict[key] 
      
        return feature2D_tensor,torch.tensor(label, dtype=torch.long)

def get_pro_fea(label):
    df=pd.read_excel('.//data//pro_fea.xlsx')
    #label='piperazine'
    X = df['Uniprot']
    y = df[label]
    data = pd.concat([X, y], axis=1)
    data_unique = data.drop_duplicates()
    ran=data_unique.shape[0]
    for index in range(0,ran-1):
        if data_unique.iloc[index,0]==data_unique.iloc[index+1,0]:
            data_unique.iloc[index,1]=1
            data_unique.iloc[index+1,1]=1
    data = data_unique.drop_duplicates()
        
    negative_samples = len(data[data[label] == 0])
    positive_samples = len(data[data[label] == 1])
    pnr=positive_samples/negative_samples

    return data,pnr

def apply_augmentation(feature2D_tensor):
    size=random.randint(100, feature2D_tensor.shape[1])
    endline=feature2D_tensor.shape[1]-size
    begin=random.randint(0, endline) 
    end=begin+size
    augmented_feature2D_tensor =feature2D_tensor[begin:end, begin:end]
    return augmented_feature2D_tensor

def data_aug(label,n):
    data,pnr=get_pro_fea(label)
    seqContactDict = {}
    for _,row in data.iterrows():
        uniprot=row['Uniprot']
        y=row[label]
        contactmap_np = np.load(f'./data/contactmap/{uniprot}.npy')
        feature2D_tensor = torch.tensor(contactmap_np, dtype=torch.float32)
        if label==1:
            num_augmentations = n
        else:
            num_augmentations = int(pnr*n)
        feature2D_tensor0 = feature2D_tensor.unsqueeze(0)
        seqContactDict[feature2D_tensor0] = y
        for _ in range(num_augmentations):
                feature2D_tensor_aug = apply_augmentation(feature2D_tensor)
                feature2D_tensor_aug = feature2D_tensor_aug.unsqueeze(0)
                seqContactDict[feature2D_tensor_aug] = y
    
    m=data.shape[0]
    train_indices, test_indices = train_test_split(range(m), test_size=0.2, random_state=42)
    dataset = ProDataset(seqContactDict)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(dataset = test_dataset,batch_size=1, shuffle=True,drop_last = True)
    train_loader = DataLoader(dataset = train_dataset,batch_size=1, shuffle=True,drop_last = True)

    return train_loader,test_loader





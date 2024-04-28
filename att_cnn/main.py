from datapre import *
from train import *
from model import *
import torch
import warnings
warnings.filterwarnings("ignore")

label='amide' #label name
n=2 #Augmentation multiple
model=att_cnn(block = ResidualBlock)#.cuda()
train_loader,test_loader=data_aug(label,n)

def main(model,epochs,lr,train_loader,test_loader,doTest=True):
    losses,accs,testResults = train(model,epochs,lr,train_loader,test_loader)

if __name__ == "__main__":
    main(model,epochs=50,lr=0.002,train_loader=train_loader,test_loader=train_loader)
import numpy as np
from sklearn import metrics
import torch
from model import *


def test(test_loader,model,criterion):
    attention_model = model
    losses = []
    accuracy = []

    print('test begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        for batch_idx, (contactmap,y) in enumerate(test_loader):  
            #contactmap, y = contactmap.cuda(), y.cuda()
            y_pred, att = attention_model(contactmap)
            if not bool(attention_model.type) :
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
                all_pred=np.concatenate((all_pred,y_pred.data.cpu().squeeze(1).numpy()),axis = 0)
                all_target = np.concatenate((all_target,y.data.cpu().numpy()),axis = 0)
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            n_batches+=1
    testSize = round(len(test_loader.dataset),3)
    testAcc = round(correct.numpy()/(n_batches*test_loader.batch_size),3)
    testRecall = round(metrics.recall_score(all_target,np.round(all_pred)),3)
    testPrecision = round(metrics.precision_score(all_target,np.round(all_pred)),3)
    testAuc = round(metrics.roc_auc_score(all_target, all_pred),3)
    print("AUPR = ",metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item()/n_batches,5)
    print("test size =",testSize,"  test acc =",testAcc,"  test recall =",testRecall,"  test precision =",testPrecision,"  test auc =",testAuc,"  test loss = ",testLoss)
    print(all_pred)
    return testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target


def train(model,epochs,lr,train_loader,test_loader,doTest=True):

    losses = []
    accs = []
    testResults = {}
    attention_model = model

    for i in range(epochs):
        print("Running EPOCH",i+1)
        total_loss = 0
        n_batches = 0
        correct = 0
        optimizer = torch.optim.Adam(attention_model.parameters(),lr=lr)
        criterion = torch.nn.BCELoss()
        attention_model.train()
        for batch_idx, (contactmap,y) in enumerate(train_loader):  
            #contactmap, y = contactmap.cuda(), y.cuda()
            y_pred, att = attention_model(contactmap)

            correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward() #retain_graph=True
            #torch.nn.utils.clip_grad_norm(attention_model.parameters(),0.5)
            optimizer.step()
            n_batches+=1
                
        avg_loss = total_loss/n_batches
        acc = correct.numpy()/(len(train_loader.dataset))
        
        losses.append(avg_loss)
        accs.append(acc)
        
        print("avg_loss is",avg_loss)
        print("train ACC = ",acc)
        if(doTest):
            testresults=test(test_loader,model=attention_model,criterion=criterion)
        
    if (doTest):
        return losses, accs, testresults 
    else:
        return losses, accs



#########################################################################
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from losses import PDF_MOI,PDF_MOE,MSE
from util import seed_torch,load_NTUdataset,img2seq,seq2img
from model import MultiTaskLossWrapper
import numpy as np
import os
#########################################################################
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    ##################################################################
    for batch_idx, (data,_) in enumerate(train_loader):
        ###################################################################### 
        inputs    = data.to(device) 
        loss = model(inputs)[2]
        ###################################################
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
#########################################################################
def get_features(model, data_loader,device):
    model.eval()
    features,labels= [],[]
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            target = target.to(device)
            inputs = data.to(device)
            features.append(model(inputs)[1])
            labels.append(target)
    features = torch.cat(features)
    labels   = torch.cat(labels).long()
    return features,labels
###################################################################################
def test(args, model, device, train_loader,test_loader, epoch, optimizer,savedir):
    model.eval()
    features_train,target_train = get_features(model,train_loader,device)
    features_test, target_test  = get_features(model,test_loader,device)
    # 
    norm_features_train = F.normalize(features_train,dim = -1)
    norm_features_test  = F.normalize(features_test, dim = -1)
    #
    distmat =  torch.matmul(norm_features_train.cpu(),norm_features_test.transpose(0,1).cpu()) 
    #
    Indx = torch.argmax(distmat,dim=0) 
    correct  = target_test.eq(target_train[Indx]).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('epoch {},Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,correct, len(test_loader.dataset), accuracy))

    return accuracy
#######################################################################    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Unspervised skeleton')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--datadir', type=str, default='NTU60')
    parser.add_argument('--evaluation', type=str, default='CS',help='evaluation (default: CS)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001)')
    parser.add_argument('--LR_STEP', type=str, default='80', metavar='LS',help='LR_STEP (default: 100)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')  
    parser.add_argument('--save_dir', type=str, default='PDF_G_A')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    ##############
    parser.add_argument('--gpuid', type=int, default=2,help='useing cuda devices id')
    parser.add_argument('--en_layers', type=int, default=1,help='encoder layers')
    parser.add_argument('--de_layers', type=int, default=1,help='decoder layers')
    parser.add_argument('--decoder_num', type=int, default=2,help='loss mode')
    parser.add_argument('--beta', type=float, default=0.3, help='decay rate for learning rate') 
    #############################################################
    args = parser.parse_args()
    print("%s_%s, gpuid %d:"%(args.datadir,args.evaluation,args.gpuid)) 
    seed_torch(args.seed)
    device = torch.device('cuda',args.gpuid)
    savedir = './checkpoint/%s/%s%s'%(args.datadir,args.evaluation,args.save_dir)    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    ####################################################################################
    # load dataset
    traindsets = load_NTUdataset(datadir=args.datadir,mode='train',evaluation = args.evaluation)
    train_loader =  torch.utils.data.DataLoader(dataset=traindsets,num_workers=16,
                                                batch_size= args.batch_size,shuffle=True)

    testdsets = load_NTUdataset(datadir=args.datadir,mode='val',evaluation = args.evaluation)
    test_loader =  torch.utils.data.DataLoader(dataset=testdsets,num_workers=16,
                                                batch_size= args.batch_size,shuffle=False)  
    #                                     
    ####################################################################################
    model = MultiTaskLossWrapper(decoder_num= args.decoder_num,en_layers=args.en_layers,de_layers=args.de_layers,beta=args.beta).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.LR_STEP, args.lr_decay_rate)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        lr_scheduler.step()
        if epoch%10 ==0:
            accuracy = test(args, model, device, train_loader,test_loader, epoch, optimizer,savedir)
 
    torch.save(model, os.path.join(savedir, 'model_%d.pth'%(epoch)))
    print("data %s _evaluation %s, gpuid %s,beta %s, accuracy %.3f:"%(args.datadir,args.evaluation,args.gpuid,args.beta,accuracy)) 

if __name__ == '__main__':
    main()

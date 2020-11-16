from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import util 
from losses import PDF_MI,PDF_OI,Pose_M,Pose_O,MSE
####################################################################################################                               
class LSTMencoder(nn.Module):
    def __init__(self,input_dim = 150,features_dim=256,num_layers = 2):
        super(LSTMencoder, self).__init__()
        #######################################################################################
        self.encoder    = nn.LSTM(input_size = input_dim,hidden_size = features_dim,
                                    num_layers = num_layers,batch_first= True,bidirectional=False)
        self.features_dim = features_dim 
    ################################################################################
    def forward(self, img):
        _,(features,_) =  self.encoder(util.img2seq(img)) 
        return features.mean(dim=0).reshape(-1,self.features_dim) 
################################################################################                             
class LSTMdecoder(nn.Module):
    def __init__(self,input_dim = 256,output_dim=150,seqlen = 50,num_layers=2):
        super(LSTMdecoder, self).__init__()
        self.decoder = nn.LSTM(input_size = input_dim,hidden_size = input_dim, 
                                num_layers = num_layers,batch_first= True) 
        self.seqlen  = seqlen
        self.dense_layers = nn.Linear(input_dim,output_dim)
    ################################################################################
    def forward(self, features):  
        seq = self.dense_layers(self.decoder(features.unsqueeze(1).repeat(1,self.seqlen,1))[0])  
        return util.seq2img(seq)
##################################################################################
class LSTMAENet(nn.Module):
    def __init__(self,input_dim = 150,features_dim = 256,en_layers = 1,de_layers = 1,decoder_num = 2):
        super(LSTMAENet, self).__init__()
        self.encoder  = LSTMencoder(input_dim = input_dim,features_dim = features_dim,num_layers = en_layers)
        decoder       = [LSTMdecoder(input_dim = features_dim,output_dim = input_dim,num_layers = de_layers) for i in range(decoder_num)]
        self.decoder  = nn.ModuleList(decoder)  
    def forward(self, img): 
        features  = self.encoder(img)  
        outputs   = [self.decoder[i](features) for i in range(len(self.decoder))]
        return outputs,features
############################################################################################################
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, input_dim = 150,features_dim = 256,en_layers = 2,de_layers = 1,decoder_num=2,beta = 0.3):
        super(MultiTaskLossWrapper, self).__init__()
        self.model    = LSTMAENet(input_dim    = input_dim,
                                  features_dim = features_dim,
                                  en_layers    = en_layers,
                                  de_layers    = de_layers,
                                  decoder_num  = decoder_num)
        self.log_vars   = nn.Parameter(torch.zeros((decoder_num)))
        self.beta   = beta
    def loss_fun(self,outputs,target):
        lossA   = torch.exp(-self.log_vars[0]) * (PDF_MI(outputs[0],target) + self.beta * Pose_M(outputs[0],target)) + self.log_vars[0]
        lossB   = torch.exp(-self.log_vars[1]) * (PDF_OI(outputs[1],target) + self.beta * Pose_O(outputs[1],target)) + self.log_vars[1]
        return lossA + lossB

    def forward(self,target): 
        outputs,features = self.model(target)
        loss   = self.loss_fun(outputs,target)
        return outputs,features,loss
##########################################################################################
if __name__ == '__main__':
    inputs = torch.rand(16,3,50,50) 
    model = LSTMAENet()  
    outputs,features = model(inputs)
    print('features size',features.size()) 
    print('outputs size',outputs[0].size())

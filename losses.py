from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import numpy as np
###################################################################################
def MSE(output,target): 
    return F.mse_loss(output,target)
###################################################################################
#Pos Flow Loss with frame difference
def PF_DF(output,target,num=5):
    FDflow_target = target[:,:,num:,:]  - target[:,:,:-num,:]
    FDflow_outpus = output[:,:,num:,:] - output[:,:,:-num,:]
    return MSE(FDflow_outpus,FDflow_target)
###################################################################################
#Pos Flow Loss with reference frame
def PF_RF(output,target): 
    Refflow_target = target - target.mean(dim=2,keepdim=True) 
    return MSE(output,Refflow_target)
###################################################################################
#Pose amplitude Decomposition flow Loss
def PDF_ME(output,target): 
    target_flow  = target - target.mean(dim=2,keepdim=True)
    return MSE(output.norm(dim=1),target_flow.norm(dim=1)) 
def PDF_MI(output,target): 
    target_flow  = target - target.mean(dim=2,keepdim=True) 
    output_flow  = output - output.mean(dim=2,keepdim=True)
    return MSE(output_flow.norm(dim=1),target_flow.norm(dim=1)) 
# Pose driection Decomposition flow Loss
def PDF_OE(output,target): 
    target_flow  = target - target.mean(dim=2,keepdim=True)
    return 1.0 - F.cosine_similarity(output,target_flow,dim=1).mean() 
def PDF_OI(output,target): 
    target_flow  = target - target.mean(dim=2,keepdim=True)
    output_flow  = output - output.mean(dim=2,keepdim=True)
    return 1.0 - F.cosine_similarity(output_flow,target_flow,dim=1).mean()
####################################################################v
def Pose_M(output,target): 
    return MSE(output.norm(dim=1),target.norm(dim=1)) 
def Pose_O(output,target): 
    return 1.0 - F.cosine_similarity(output,target,dim=1).mean()
###################################################################################
# Pose Magnitude and Orientation Decomposition flow Loss,inputs is 2
def PDF_MOE(outputs,target): 
    return PDF_ME(outputs[0],target) + \
           PDF_OE(outputs[1],target)
##########################################################
def PDF_MOI(outputs,target): 
    return PDF_MI(outputs[0],target)     + PDF_OI(outputs[1],target) + \
           1.0 * MSE(outputs[0],outputs[1])
#########################################################################################
def PDF_losses(outputs,target,loss_fun):
    return globals()[loss_fun](outputs[0] if len(outputs) ==1 else outputs,target)

# def PDF_losses(outputs,target,lossmode='MSE'): 
#     if   lossmode == 'MSE':
#         loss = MSEloss(outputs[0],target)
#     #帧差
#     elif lossmode == 'PFDF':
#         loss = Pose_FDFlow(outputs[0],target)
#     #参考帧帧差
#     elif lossmode == 'PRF':
#         loss = Pose_RFlow(outputs[0],target)
#     #光流幅度-直接
#     elif lossmode == 'PDF_Magnitude_E':  
#         loss = PDF_Magnitude_E(outputs[0],target)
#     #光流幅度-间接
#     elif lossmode == 'PDF_Magnitude_I':  
#         loss = PDF_Magnitude_I(outputs[0],target)
#     #光流方向-直接     
#     elif lossmode == 'PDF_Orientation_E':
#         loss = PDF_Orientation(outputs[0],target,is_Dout = is_Dout)
#     #光流方向+幅度，如果不是直接输出，引入两个分支之间的mse约束      
#     elif lossmode == 'PDF_MO':
#         loss = PDF_MO(outputs,target,is_Dout = is_Dout) 
#     elif lossmode == 'PDF_MMO':
#         loss = PDF_MMO(outputs,target,is_Dout = is_Dout)
#     else:
#         print('No the type loss!')
#     return loss 


    # target_flow = target  - target.mean(dim=2,keepdim=True)
    # target_flow = target_flow/(target_flow.norm(dim=1).unsqueeze(dim=1) + 1e-10)
    # output_flow = output - output.mean(dim=2,keepdim=True)
    # output_flow = output_flow/(output_flow.norm(dim=1).unsqueeze(dim=1) + 1e-10)
    # output_flow = output if is_Dout else output/(output.norm(dim=1).unsqueeze(dim=1) + 1e-10)
    # return F.mse_loss(output_flow,target_flow)



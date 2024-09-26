# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:51:56 2024

@author: shach
"""

########
##Kalmanet
#########

import torch
import torch.nn as nn
import math
from datetime import datetime
#import Filters.EKF_test_withbias_IK as EKF_test
import Filters.EKF_test_withbias_IK as EKF_test
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Simulations.utils import Short_Traj_Split
from Simulations.Raven_ii_matlab.parameters_withbias_trainIK import m1x_0, m2x_0, m, n, f, h, fInacc, Q_structure, R_structure
from KNet.KalmanNet_nn_withbias_IK import InverseKinematicsNN
from Pipelines.Pipeline_EKF_withbias_IK import Pipeline_EKF
from KNet.KalmanNet_nn_withbias_IK import KalmanNetNN
from Plot import Plot_extended as Plot
import torch.nn.functional as F

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 600
args.N_CV = 50
args.N_T = 60
args.T = 100
args.T_test = 100
### training parameters
args.use_cuda = False# use GPU or not
args.n_steps = 150
args.n_batch = 4 #used to be 128
args.lr = 1e-4#used to be 1e-5
args.wd = 1e-2
## Flags ##
plot_bias = False
EKF = False
if args.use_cuda:
    if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
    else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop the dataset sequences into smaller ones
path_results = 'KNet/'
DatafolderName = 'Simulations/Raven_ii_matlab/data/'
DatafileName = 'processed_trajectories_devide.pt'

r = torch.tensor([1])
lambda_q = torch.tensor([0.3873])

print("1/r2 [dB]: ", 10 * torch.log10(1/r[0]**2))
print("Search 1/q2 [dB]: ", 10 * torch.log10(1/lambda_q[0]**2))
# Q = 0.0000001 * Q_structure
# R = 1000000 * R_structure 
Q = 0.1* Q_structure
R = 10*R_structure 




##########################################
### Load and prepare pre-generated data ##
##########################################
# Load the pre-generated data
directory = r'C:\Users\shach\Documents\shachar\project_code\python\KalmanNet_TSP-main\Simulations\Raven_ii_matlab\data'
data = torch.load(os.path.join(directory,'processed_trajectories_710on100.pt'), map_location=device)
obs_reshaped = data['observations']
states_reshaped = data['states']
print(f'Data size {obs_reshaped.shape}')
states_reshaped1 = np.array(states_reshaped)
# Generate a permutation of indices for the first dimension
permutation = np.random.permutation(obs_reshaped.shape[0])

# Split the data into training, validation, and test sets
total_traj = obs_reshaped.size(0)
train_traj = args.N_E
cv_traj = args.N_CV
test_traj = args.N_T
train_input = obs_reshaped[:train_traj]
train_target = states_reshaped[:train_traj]
cv_input = obs_reshaped[train_traj:train_traj + cv_traj]
cv_target = states_reshaped[train_traj:train_traj + cv_traj]
test_input = obs_reshaped[train_traj + cv_traj:]
test_target = states_reshaped[train_traj + cv_traj:]
# Save the dataset
torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target], DatafolderName + DatafileName)

#intiate m1 and m2
m1x_0 = test_target[:,:,0]
m2x_0 = Q
# True Model
sys_model_true = SystemModel(f, Q, h, R, args.T, args.T_test,m,n)
sys_model_true.InitSequence(m1x_0, m2x_0)

###check that there is data
def check_for_nans(data, name):
    if torch.isnan(data).any():
        print(f"NaNs found in {name}")
    else:
        print(f"No NaNs in {name}")

check_for_nans(train_input, "train_input")
check_for_nans(train_target, "train_target")
check_for_nans(cv_input, "cv_input")
check_for_nans(cv_target, "cv_target")
check_for_nans(test_input, "test_input")
check_for_nans(test_target, "test_target")

#########################
print("Data Load")
#########################
[train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName + DatafileName, map_location=device)
print("load dataset to device:", train_input.device)
print("testset size:", test_target.size())
print("trainset size:", train_target.size())
print("cvset size:", cv_target.size())

########################################
### Evaluate Observation Noise Floor ###
########################################
args.N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]
for j in range(0, args.N_T):        
    MSE_obs_linear_arr[j] = loss_obs(test_input[j], test_target[j,:3,:]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")
###################################################
args.N_E = len(train_input)
MSE_obs_linear_arr = torch.empty(args.N_E)# MSE [Linear]
for j in range(0, args.N_E):        
    MSE_obs_linear_arr[j] = loss_obs(train_input[j], train_target[j,:3,:]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(train dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(train dataset) - STD:", obs_std_dB, "[dB]")

########################
### Evaluate Filters ###
########################

if EKF:
    batch_size = args.n_batch
    num_epochs = 1

    # Initialize the IK Model
    IK_model = InverseKinematicsNN()

    # If using CUDA
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    IK_model = IK_model.to(device)

    # Optimizer for training the IK model
    optimizer = torch.optim.Adam(IK_model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # To store the concatenated outputs across all trajectories
    all_EKF_out = []
    all_MSE = []

    # Training loop for epochs
    for epoch in range(num_epochs):
        

        # Shuffle the dataset at the beginning of each epoch if needed
        permutation = torch.randperm(test_input.size(0))
        test_input_shuffled = test_input[permutation]
        test_target_shuffled = test_target[permutation]

        # Iterate over batches
        for i in range(0, test_input.size(0), batch_size):
            
            # Get the batch
            batch_input = test_input_shuffled[i:i+batch_size].to(device)
            batch_target = test_target_shuffled[i:i+batch_size].to(device)
            
            # Initialize m1x_0 and m2x_0 for the batch (based on your existing initialization logic)
            m1x_0_new = m1x_0[i , :]  # Assuming m1x_0 is the initial state for each sample in the batch
            
            # Set up initial conditions in SysModel
            sys_model_true.m1x_0 = m1x_0_new

            # Run EKF and train the IK model for this batch
            [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out] = EKF_test.EKFTest(
                args, sys_model_true, IK_model, batch_input, batch_target, optimizer
            )

            # Append the EKF_out for the current batch to the list
            all_EKF_out.append(EKF_out)
            all_MSE.append(MSE_EKF_dB_avg)

    # Concatenate all EKF_out tensors along the batch dimension
    final_EKF_out = torch.cat(all_EKF_out, dim=0)
    EKF_out = final_EKF_out
    print(f'EKF MSE is {np.mean(all_MSE)}[db]')
    print(f'EKF out shape: {final_EKF_out.shape}')

# Extract the first elements from the third dimension
test_init = test_target[:, :, 0]  
train_init = train_target[:, :, 0]  
cv_init = cv_target[:, :, 0]  

#  ensure correctness of m1
m1x_0 = train_target[:,:,0]
sys_model_true.m1x_0=m1x_0
cv_init=cv_target[:,:,0]
train_init=train_target[:,:,0]
########################################
### KalmanNet with model mismatch ######
########################################
## Build Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model_true, args)
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setssModel(sys_model_true)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
# Train Neural Network
KNet_Pipeline.setTrainingParams(args)
if(chop):
    KNet_Pipeline.NNTrain(sys_model_true,cv_input,cv_target,train_input,train_target,path_results,\
                          randomInit=True,train_init=train_init)
else:
    KNet_Pipeline.NNTrain(sys_model_true,cv_input,cv_target,train_input,train_target,path_results)
# Test Neural Network
m1x_0 = test_target[:,:,0]
sys_model_true.m1x_0=m1x_0
sys_model_true.InitSequence(m1x_0, m2x_0)
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out,t] = KNet_Pipeline.NNTest(sys_model_true,test_input,test_target,path_results,MaskOnState=False, randomInit=False)
is_equal = torch.equal(knet_out[:,:,0], test_target[:, :, 0])

if not EKF:
    EKF_out = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Save trajectories
trajfolderName = 'Simulations/Raven_ii_matlab' + '/'
DataResultName = 'traj_Raven_II_all'

# Convert PyTorch tensors to NumPy arrays and detach
data_np = {
    'True': test_target.detach().numpy(),
    'Observation': test_input.detach().numpy(),
    'EKF':EKF_out.detach().numpy(),
    #'EKF_partial':EKF_out.detach().numpy(),
    'KNet': knet_out.detach().numpy()
}

# Save the data to a .mat file
scipy.io.savemat(trajfolderName+DataResultName+'EKF.mat', data_np)

target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
torch.save({
            'True':target_sample,
            'Observation':input_sample,
            'EKF':EKF_out,
            'EKF_partial':EKF_out,                       
            'KNet': knet_out,
            }, trajfolderName+DataResultName)











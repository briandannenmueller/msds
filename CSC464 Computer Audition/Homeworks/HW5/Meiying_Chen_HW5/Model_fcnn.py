from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os
import numpy as np
import Data
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import argparse
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm, trange



class ModelSingleStep(torch.nn.Module):
    def __init__(self, blockSize):
        super(ModelSingleStep, self).__init__()
        self.blockSize = blockSize

        ###################################
        #define your layers here
        ###################################
        self.fc1 = torch.nn.Linear(2049, 1000)
        self.fc2 = torch.nn.Linear(1000, 400)
        self.fc3 = torch.nn.Linear(400, 1000) 
        self.fc4 = torch.nn.Linear(1000, 2049)  
        ###################################

        self.initParams()

    def initParams(self):
        for param in self.parameters():
             if len(param.shape)>1:
                 torch.nn.init.xavier_normal_(param)


    def encode(self, x):
        ###################################
        #implement the encoder
        ###################################
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        h = x
        ###################################
        return h

    def decode(self, h):
        ###################################
        #implement the decoder
        ###################################
        h = F.leaky_relu(self.fc3(h))
        o = torch.sigmoid(self.fc4(h))
        ###################################
        return o

    def forward(self, x):
        #glue the encoder and the decoder together
        h = self.encode(x)
        o = self.decode(h)
        return o

    def process(self, magnitude):
        #process the whole chunk of spectrogram at run time
        result= magnitude.copy()
        with torch.no_grad():
            nFrame = magnitude.shape[1]
            for i in range(nFrame):
                result[:,i] = magnitude[:,i]*self.forward(torch.from_numpy(magnitude[:,i].reshape(1,-1))).numpy()
        return result 

def validate(model, dataloader):
    model.eval()
    with torch.no_grad():
    #Each time fetch a batch of samples from the dataloader
        for sample in dataloader:
    ######################################################################################
    # Implement here your validation loop. It should be similar to your train loop
    # without the backpropagation steps
    ######################################################################################
            model.zero_grad()
            mixture = sample['mixture']
            target = sample['vocal']
            seqLen = mixture.shape[2]
            winLen = mixture.shape[1]
            currentBatchSize = mixture.shape[0]
            result = torch.zeros((winLen, seqLen), dtype=torch.float32)
            outputs = model(mixture)
            criterion = nn.KLDivLoss()
            validationLoss = criterion(outputs, target)
    ######################################################################################

    model.train()
    return validationLoss

def saveFigure(result, target, mixture):
    plt.subplot(3,1,1)
    plt.pcolormesh(np.log(1e-4+result), vmin=-300/20, vmax = 10/20)
    plt.title('estimated')

    plt.subplot(3,1,2)
    plt.pcolormesh(np.log(1e-4+target.cpu()[0,:,:].numpy()), vmin=-300/20, vmax =10/20)
    plt.title('vocal')
    plt.subplot(3,1,3)

    plt.pcolormesh(np.log(1e-4+mixture.cpu()[0,:,:].numpy()), vmin=-300/20, vmax = 10/20)
    plt.title('mixture')

    plt.savefig("result_feedforward.png")
    plt.gcf().clear()

if __name__ == "__main__":
    ######################################################################################
    # Load Args and Params
    ######################################################################################
    parser = argparse.ArgumentParser(description='Train Arguments')
    parser.add_argument("--blockSize", type=int, default = 4096)
    parser.add_argument('--hopSize', type=int, default = 2048)
    # how many audio files to process fetched at each time, modify it if OOM error
    parser.add_argument('--batchSize', type=int, default = 4)
    # set the learning rate, default value is 0.0001
    parser.add_argument('--lr', type=float, default=1e-4)
    # Path to the dataset, modify it accordingly
    parser.add_argument('--dataset', type=str, default = '/home/melissa/Data/DSD100/')
    # set --load to 1, if you want to restore weights from a previous trained model
    parser.add_argument('--load', type=int, default = 0)
    # path of the checkpoint that you want to restore
    parser.add_argument('--checkpoint', type=str, default = 'savedModel_feedForward_best.pt')
    
    parser.add_argument('--seed', type=int, default = 555)
    args = parser.parse_args()

    # Random seeds, for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fs = 32000
    blockSize = args.blockSize
    hopSize = args.hopSize
    PATH_DATASET = args.dataset
    batchSize= args.batchSize
    minValLoss = np.inf

    # transformation pipeline for training data
    transformTrain = transforms.Compose([
        #Randomly rescale the training data
        Data.Transforms.Rescale(0.8, 1.2),

        #Randomly shift the beginning of the training data, because we always do chunking for training in this case
        Data.Transforms.RandomShift(fs*30),

        #transform the raw audio into spectrogram
        Data.Transforms.MakeMagnitudeSpectrum(blockSize = blockSize, hopSize = hopSize),

        #shuffle all frames of a song for training the single-frame model , remove this line for training a temporal sequence model
        Data.Transforms.ShuffleFrameOrder()
        ])

    # transformation pipeline for training data. Here, we don't have to use any augmentation/regularization techqniques
    transformVal = transforms.Compose([
        #transform the raw audio into spectrogram
        Data.Transforms.MakeMagnitudeSpectrum(blockSize = blockSize, hopSize = hopSize),
        ])

    #initialize dataloaders for training and validation data, every sample loaded will go thourgh the preprocessing pipeline defined by the above transformations
    #workers will restart after each epoch, which takes a lot of time. repetition = 8  repeats the dataset 8 times in order to reduce the waiting time
    # so, in this case,  1 epoch is equal to 8 epochs. For validation data, there is not point in repeating the dataset.
    datasetTrain = Data.DSD100Dataset(PATH_DATASET, split = 'Train', mono =True, transform = transformTrain, repetition = 8)
    datasetValid = Data.DSD100Dataset(PATH_DATASET, split = 'Valid', mono =True, transform = transformVal, repetition = 1)

    #initialize the data loader
    #num_workers means how many workers are used to prefetch the data, reduce num_workers if OOM error
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size = batchSize, shuffle=True, num_workers = 4, collate_fn = Data.collate_fn)
    dataloaderValid = torch.utils.data.DataLoader(datasetValid, batch_size = 10, shuffle=False, num_workers = 0, collate_fn = Data.collate_fn)

    #initialize the Model
    model = ModelSingleStep(blockSize)

    # if you want to restore your previous saved model, set --load argument to 1
    if args.load == 1:
        checkpoint = torch.load(args.checkpoint)
        minValLoss = checkpoint['minValLoss']
        model.load_state_dict(checkpoint['state_dict'])

    #determine if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #initialize the optimizer for paramters
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    model.train(mode=True)
  
    lossMovingAveraged  = -1
    
    #################################### 
    #The main loop of training
    #################################### 
    for epoc in range(100):
        iterator = iter(dataloaderTrain)
        with trange(len(dataloaderTrain)) as t:
            for idx in t:
                #Each time fetch a batch of samples from the dataloader
                sample = next(iterator)
                #the progress of training in the current epoch

                #Remember to clear the accumulated gradient each time you perfrom optimizer.step()
                model.zero_grad()

                #read the input and the fitting target into the device
                mixture = sample['mixture'].to(device)
                target = sample['vocal'].to(device)
                
                seqLen = mixture.shape[2]
                winLen = mixture.shape[1]
                currentBatchSize = mixture.shape[0]

                #store the result for the first one for debugging purpose
                result = torch.zeros((winLen, seqLen), dtype=torch.float32)

				#################################
                # Fill the rest of the code here#
				#################################
				# forward, backward, optimize

                #You can use mixture[:, :, stepBegin: stepEnd] to assess a range of frames
                for i in  range(seqLen):
                    estimated = model.forward(mixture[:,:,i])
                    x = target[:,:,i]
                    y = estimated[:,:]*mixture[:,:,i]
                    # loss = torch.sum((x+1e-6)*(torch.log(x+1e-6)-torch.log(y+1e-6))-x+y)
                    loss = F.mse_loss(estimated, target)
                    loss.backward()
                    optimizer.step()
               
                # store your smoothed loss here
                eta = 0.99
                lossMovingAveraged = eta*lossMovingAveraged + (1-eta)*loss
                # this is used to set a description in the tqdm progress bar 
                t.set_description(f"epoc : {epoc}, loss {lossMovingAveraged}")
                #save the model

            # plot the first one in the batch for debuging purpose
            saveFigure(result, target, mixture)

        # create a checkpoint of the current state of training
        checkpoint = {
                    'state_dict': model.state_dict(),
                    'minValLoss': minValLoss,
                     }
        # save the last checkpoint
        torch.save(checkpoint, 'savedModel_feedForward_last.pt')

        #### Calculate validation loss
        valLoss = validate(model, dataloaderValid)
        print(f"validation Loss = {valLoss:.4f}")
        
        if valLoss < minValLoss:
            minValLoss = valLoss
            # then save checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'minValLoss': minValLoss,
                    }
            torch.save(checkpoint, 'savedModel_feedForward_best.pt')



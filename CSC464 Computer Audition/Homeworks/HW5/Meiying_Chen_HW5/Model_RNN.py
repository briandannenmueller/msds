


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


from layers import *


class ModelSingleStep(torch.nn.Module):
    def __init__(self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=1487,
        unidirectional=False,
        power=1):
        super(ModelSingleStep, self).__init__()

        ###################################
        #define your layers here
        ###################################

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )


        ###################################

        self.initParams()

    def initParams(self):
        for param in self.parameters():
             if len(param.shape)>1:
                 torch.nn.init.xavier_normal_(param)



    def forward(self, x):
        #glue the encoder and the decoder together
        # h = self.encode(x)
        # x = self.decode(h)
        x = torch.stack([x,x], dim=-2).permute(3, 0, 2, 1)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        h = torch.tanh(x)


        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        x = x.mean(dim=-2).permute(1, 2, 0)

        return x

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
    validationLoss = 0.0
    with torch.no_grad():
    #Each time fetch a batch of samples from the dataloader
        for sample in dataloader:

    ######################################################################################
    # Implement here your validation loop. It should be similar to your train loop
    # without the backpropagation steps
    ######################################################################################

            model.zero_grad()
            mixture = sample['mixture'].to(device)
            target = sample['vocal'].to(device)
            estimated = model(mixture)
            validationLoss = F.mse_loss(estimated, target)


    model.train()
    return validationLoss/len(dataloader)

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
    parser.add_argument('--dataset', type=str, default = '/home/melissa/Data/DSD100')
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
    model = ModelSingleStep()
    model.load_state_dict(torch.load("vocals.pth"))

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
                estimated = model(mixture)
                loss = F.mse_loss(estimated, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # store your smoothed loss here #############
                eta = 0.99
                if lossMovingAveraged == -1:
                    lossMovingAveraged = loss
                else:
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
       #print(f"validation Loss = {valLoss:.4f}")
        print('validation Loss =', valLoss)

        if valLoss < minValLoss:
            minValLoss = valLoss
            # then save checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'minValLoss': minValLoss,
                    }
            torch.save(checkpoint, 'savedModel_feedForward_best.pt')



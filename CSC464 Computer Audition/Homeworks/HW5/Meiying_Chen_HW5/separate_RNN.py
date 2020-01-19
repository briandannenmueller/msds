

from __future__ import print_function, division, absolute_import, unicode_literals
import six
import argparse
import os
import numpy as np
import Model_RNN as Model
import torch
import util
import sys


if __name__=="__main__":
    blockSize = 4096
    hopSize = 2048

    if len(sys.argv) != 3:
        print("Usage:\n", sys.argv[0], "input_path output_path")
        exit(1)

    #read the wav file
    x, fs = util.wavread(sys.argv[1])
    #downmix to single channel
    x = np.mean(x,axis=-1)
    #perform stft
    S = util.stft_real(x, blockSize=blockSize, hopSize=hopSize)
    magnitude = np.abs(S).astype(np.float32)
    angle = np.angle(S).astype(np.float32)

    #initialize the model
    model = Model.ModelSingleStep(blockSize)

    #load the pretrained model
    checkpoint = torch.load("savedModel_RNN_best.pt", map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['state_dict'])

    #switch to eval mode
    model.eval()
    # magnitude = torch.Tensor(magnitude).view(1, magnitude.shape[0], magnitude.shape[1])


    ###################################
    #Run your Model here to obtain a mask
    ###################################
    magnitude = torch.from_numpy(magnitude.reshape((1,magnitude.shape[0],magnitude.shape[1]))) 
    with torch.no_grad():
        magnitude_masked = model.forward(magnitude).numpy().T
    ###################################


    #perform reconstruction
    y = util.istft_real(magnitude_masked * np.exp(1j* angle), blockSize=blockSize, hopSize=hopSize)

    #save the result
    util.wavwrite(sys.argv[2], y,fs)

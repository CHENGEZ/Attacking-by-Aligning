import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import sys

class AsrPlt():
    def __init__(self, log_dir):
        self.log_dir    = log_dir
        self.asr     = []
        self.writer     = SummaryWriter(self.log_dir)
       

    def append_asr(self, epoch, asr):

        self.asr.append(asr)
        
        with open(os.path.join(self.log_dir, "epoch_asr.txt"), 'a') as f:
            f.write(str(asr))
            f.write("\n")
        self.writer.add_scalar('asr', asr, epoch)
        self.asr_plot()

    def asr_plot(self):
        iters = range(len(self.asr))

        plt.figure()
        plt.plot(iters, self.asr, 'red', linewidth = 2, label='ASR')
        try:
            if len(self.asr) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.asr, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth ASR')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Attack Success Rate')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_asr.png"))

        plt.cla()
        plt.close("all")


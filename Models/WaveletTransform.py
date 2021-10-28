# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import pickle
from torch.autograd import Variable

class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True, channel=3):
        super(WaveletTransform, self).__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = channel * ks * ks

        self.channel = channel
        if dec:
            self.conv = nn.Conv2d(in_channels=channel, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=channel,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=channel, kernel_size=ks, stride=ks, padding=0,
                                           groups=channel, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                dct = pickle.load(f,encoding='latin1')
                f.close()
                K = torch.from_numpy(dct['rec%d' % ks])
                kernelW = K[0:4,:,:,:]
                weightlist = []
                times = nc//4
                for i in  range(times):
                    newW = kernelW.clone()
                    weightlist.append(newW)

                K = torch.cat(weightlist,dim=0)

                m.weight.data = K
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], self.channel, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, self.channel, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output


if __name__ == '__main__':
    pass
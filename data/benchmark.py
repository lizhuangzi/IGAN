import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]

            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            filename = filename.split('HR_')[1]
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/X{}down_{}{}'.format(s, s, filename, self.ext)
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, '8XHR')
        #self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'BicDown')
        self.ext = '.png'

    # def _get_patch(self, lr, hr):
    #     patch_size = self.args.patch_size
    #     scale = self.scale[self.idx_scale]
    #     multi_scale = len(self.scale) > 1
    #     if self.train:
    #         lr, hr = common.get_patch(
    #             lr, hr, patch_size, scale, multi_scale=multi_scale
    #         )
    #         lr, hr = common.augment([lr, hr])
    #         lr = common.add_noise(lr, self.args.noise)
    #     else:
    #         lr = common.add_noise(lr, self.args.noise)
    #         ih, iw = lr.shape[0:2]
    #         hr = hr[0:ih * scale, 0:iw * scale]
    #
    #     return lr, hr
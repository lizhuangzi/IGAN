import os
import math
import time
import datetime
from functools import reduce

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
import os
import pytorch_ssim

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.':
                args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        #trainer.loss.save(self.dir)
        #trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_psnr(epoch)
        # torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        # torch.save(
        #     trainer.optimizer.state_dict(),
        #     os.path.join(self.dir, 'optimizer.pt')
        # )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = './resultlist/results/{}_x{}_'.format(filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imsave('{}{}.png'.format(filename, p), ndarr)

    def save_results2(self, filename, save_list1,save_list2, scale):
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list1, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imsave('{}{}.png'.format(filename, p), ndarr)
            #imageio.imsave('{}{}bicubic.png'.format(filename, p), ndarr)
        for v, p in zip(save_list2, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            #imageio.imsave('{}{}.png'.format(filename, p), ndarr)
            imageio.imsave('{}{}bicubic.png'.format(filename, p), ndarr)

    def save_SRresults(self, filename, save_list, scale,modelname,datasetname):
        filename = './resultlist/{}/{}/{}_x{}_'.format(modelname,datasetname,filename, scale)
        postfix = ('SR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imsave('{}{}.png'.format(filename, p), ndarr)

    def save_SRresults8x(self, filename, save_list, scale,modelname,datasetname):
        filename = './resultlist8x/{}/{}/{}_x{}_'.format(modelname,datasetname,filename, scale)
        postfix = ('SR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imsave('{}{}.png'.format(filename, p), ndarr)

    def save_SRresultsParis(self, filename, save_list,modelname,categoryName):

        categorypath = os.path.join('./ClassResult',modelname,categoryName)
        if os.path.exists(categorypath)==False:
            os.mkdir(categorypath)

        filename = './ClassResult/{}/{}/{}'.format(modelname,categoryName,filename)
        postfix = ('SR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imsave('{}{}.jpg'.format(filename, p), ndarr)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)



def calc_ssim(sr,hr,scale,rgb_range):
    sr = sr.data.div(rgb_range)
    hr = hr.data.div(rgb_range)
    if hr.size(1) > 1:
        convert = hr.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064

        hr.mul_(convert).div_(256)
        sr.mul_(convert).div_(256)
        hr = hr.sum(dim=1, keepdim=True)
        sr = sr.sum(dim=1, keepdim=True)

    ssim = pytorch_ssim.ssim(hr,sr)
    ssim = ssim.cpu().numpy()
    return ssim

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    # ycbcrdiff = ConvertYCbCr(diff)
    # diff = torch.unsqueeze(ycbcrdiff[:, 0, :, :], 1)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler


def ConverttoRGB3(imgTensr):
    imgTensr = imgTensr*255
    Y = torch.unsqueeze(imgTensr[0],0)
    Cb = torch.unsqueeze(imgTensr[1],0)
    Cr = torch.unsqueeze(imgTensr[2],0)
    R = Y + 1.402*(Cr-128)
    G = Y - 0.34414*(Cb-128)-0.71414*(Cr-128)
    B = Y + 1.772*(Cb-128)
    RGBTensor = torch.cat([R,G,B],0)
    return RGBTensor/255

def ConvertYCbCr(imgTensr):
    imgTensr = imgTensr*255
    R = torch.unsqueeze(imgTensr[:,0,:,:],1)
    G = torch.unsqueeze(imgTensr[:,1,:,:],1)
    B = torch.unsqueeze(imgTensr[:,2,:,:],1)
    Y = 0.299*R+0.587*G+0.114*B
    Cb = -0.1687*R-0.3313*G+0.5*B+128
    Cr = 0.5*R-0.4187*G-0.0813*B+128
    RGBTensor = torch.cat([Y,Cb,Cr],1)
    return RGBTensor/255


def chop_forward(x, model, scale, shave=16, min_size=100000, nGPUs=1):
    b, c, h, w = x.size()
    scale1 = scale[0]
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)

            output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale1 * h, scale1 * w
    h_half, w_half = scale1 * h_half, scale1 * w_half
    h_size, w_size = scale1 * h_size, scale1 * w_size
    shave *= scale1

    datas = torch.FloatTensor(b, c, h, w)
    output = Variable(datas, volatile=True)
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

#
# def chop_forward16(x, model, scale, shave=16, min_size=100000, nGPUs=1):
#     b, c, h, w = x.size()
#     scale = scale[0]
#     h_quart, w_quart = h // 4, w // 4
#     h_size, w_size = h_quart + shave, w_quart + shave
#     inputlist = [
#
#
#         x[:, :, 0:h_size, 0:w_size],
#         x[:, :, 0:h_size, w_size:w_size],
#         x[:, :, 0:h_size, (w - w_size):w],
#         x[:, :, (h - h_size):h, 0:w_size],
#         x[:, :, (h - h_size):h, (w - w_size):w]
#
#     ]
#
#     if w_size * h_size < min_size:
#         outputlist = []
#         for i in range(0, 4, nGPUs):
#             input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
#
#             output_batch = model(input_batch)
#             outputlist.extend(output_batch.chunk(nGPUs, dim=0))
#     else:
#         outputlist = [
#             chop_forward(patch, model, scale, shave, min_size, nGPUs) \
#             for patch in inputlist]
#
#     h, w = scale * h, scale * w
#     h_half, w_half = scale * h_half, scale * w_half
#     h_size, w_size = scale * h_size, scale * w_size
#     shave *= scale
#
#     datas = torch.FloatTensor(b, c, h, w)
#     output = Variable(datas, volatile=True)
#     output[:, :, 0:h_half, 0:w_half] \
#         = outputlist[0][:, :, 0:h_half, 0:w_half]
#     output[:, :, 0:h_half, w_half:w] \
#         = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
#     output[:, :, h_half:h, 0:w_half] \
#         = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
#     output[:, :, h_half:h, w_half:w] \
#         = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
#
#     return output
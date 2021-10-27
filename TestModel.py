import os
import math
from decimal import Decimal
import pandas as pd
import utility
#import pytorch_ssim
import torch
from torch.autograd import Variable
from tqdm import tqdm
import data
import Models
from optionTest import args
import loss
import Models.Bicubic as core
os.environ["CUDA_VISIBLE_DEVICES"] = "1"





class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp  # checkpoint
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.globleepoch = 0
        self.EXPresult = {'psnr': [],'ssim': [],'trainloss': []}
        self.bestresult = 0

    def train(self):
        self.scheduler.step()
        self.loss.step()
        # epoch = self.scheduler.last_epoch + 1
        # lr = self.scheduler.get_lr()[0]
        self.globleepoch+=1

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        tqdmtrain = tqdm(self.loader_train)
        traincount = 0
        totoalloss = 0
        for batch, (lr, hr, _) in enumerate(tqdmtrain):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            tqdmtrain.set_description(desc='epoch%d loss:%f' % (self.globleepoch,(loss/batch).item()))

            totoalloss+=loss.item()
            traincount+=1

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
        train_averageloss = totoalloss/(self.args.batch_size*traincount)
        self.EXPresult['trainloss'].append(train_averageloss)



    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        scale = int(self.scale)

        timer_test = utility.timer()
        with torch.no_grad():
            eval_acc = 0
            total_ssim = 0
            count = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                filename = filename[0]
                no_eval = (hr.nelement() == 1)
                if not no_eval:
                    lr, hr = self.prepare([lr, hr])
                else:
                    lr = self.prepare([lr])[0]

                sr = self.model(lr) #100000 #sr = core.imresize(lr, scale=8)

                #sr = utility.chop_forward(lr,self.model,self.args.scale,16,100000,4)
                #sr = core.imresize(lr, scale=4)
                sr = sr.cuda()
                sr = sr.detach()
                sr = utility.quantize(sr, self.args.rgb_range)

                save_list = [sr]
                if not no_eval:
                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=False
                    )
                    #save_list.extend([lr, hr])
                    count+=1

                total_ssim += utility.calc_ssim(sr,hr,scale,self.args.rgb_range)


            averagepsnr = eval_acc*1.0/count
            averagessim = total_ssim*1.0/count
            print('PSNRis %f, SSIM is %f'%(averagepsnr,averagessim))





    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


checkpoint = utility.checkpoint(args)
loader = data.Data(args)
model = Models.Model(args, checkpoint)
print('# model parameters:', sum(param.numel() for param in model.parameters()))
loss = loss.Loss(args, checkpoint) if not args.test_only else None
t = Trainer(args, loader, model, loss, checkpoint)
t.test()
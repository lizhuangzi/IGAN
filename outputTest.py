import os
import math
from decimal import Decimal
import pandas as pd
import utility
import pytorch_ssim
import torch
from torch.autograd import Variable
from tqdm import tqdm
import data
import Models
from optionTest import args
import loss
os.environ["CUDA_VISIBLE_DEVICES"] = "0"





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
        scale = 8

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

                #sr = self.model(lr)
                sr = utility.chop_forward(lr,self.model,self.args.scale,16,100000,4)
                sr = sr.cuda()
                sr = sr.detach()
                sr = utility.quantize(sr, self.args.rgb_range)

                save_list = [sr]
                if not no_eval:
                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=False
                    )
                    save_list.extend([lr, hr])
                    count+=1

                total_ssim += utility.calc_ssim(sr,hr,scale,self.args.rgb_range)

                # save image
                self.ckp.save_results(filename, save_list, scale)


            averagepsnr = eval_acc*1.0/count
            averagessim = total_ssim*1.0/count
            print('PSNRis %f, SSIM is %f'%(averagepsnr,averagessim))
            # if self.bestresult<averagepsnr:
            #     self.bestresult = averagepsnr
            #     torch.save(self.model.state_dict(), './savedModel/TestIIA_Progess.parm')
            #
            # self.EXPresult['psnr'].append(averagepsnr)
            # self.EXPresult['ssim'].append(averagessim)
            #
            # out_path = 'statistics/'
            # data_frame = pd.DataFrame(
            #     data={'psnr': self.EXPresult['psnr'],'ssim':self.EXPresult['ssim'],'trainloss':self.EXPresult['trainloss']},
            #     index=range(1, self.globleepoch + 1))
            # data_frame.to_csv(out_path + 'TestIIA_Progess' + '4.csv', index_label='Epoch')




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

from torch.utils.data import DataLoader
from data_utils import TestDatasetFromFolder3

test_set = TestDatasetFromFolder3('/home/lizhuangzi/Desktop/test1', upscale_factor=4)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

checkpoint = utility.checkpoint(args)
model = Models.Model(args, checkpoint)
for i,(val_lr,val_bic) in enumerate(test_loader):
    val_lr = val_lr.cuda()
    val_lr = val_lr*255.0

    val_bic = val_bic*255.0
    #sr = model(val_lr)
    sr = utility.chop_forward(val_lr, model, [4] ,16, 1000000, 4)
    #sr = sr.cuda()
    sr = sr.detach()
    sr = utility.quantize(sr, 255)

    val_bic = utility.quantize(val_bic,255)


    save_list = [sr]
    save_list2 = [val_bic]

    filename = "/home/lizhuangzi/Desktop/test1/output/%d_srresult" %i
    checkpoint.save_results2(filename, save_list,save_list2, 4)

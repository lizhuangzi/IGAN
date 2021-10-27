import torch

import utility
import data
import Models
import loss
from optionIGAN import args
from trainerIGAN4 import Trainer
import os
import numpy as np
import random

checkpoint = utility.checkpoint(args)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if checkpoint.ok:
    seed = 1234
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    #torch.manual_seed(seed)
    random.seed(seed)

    loader = data.Data(args)
    model = Models.Model(args, checkpoint)
    print('# model parameters:', sum(param.numel() for param in model.parameters()))
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
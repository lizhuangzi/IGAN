from Models import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.MPNCONV.python import MPNCONV
from Models.WaveletTransform import WaveletTransform
#
def make_model(args, parent=False):
    return IIAN(args)



class MYSOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(MYSOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        #self.compressConv = nn.Conv2d(channel, 1, 1, padding=0, bias=True)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, prevousinput, x):
        x = x-prevousinput
        batch_size, C, h, w = x.shape  # x: NxCxHxW

        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = MPNCONV.CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = MPNCONV.SqrtmLayer(cov_mat,
                                         5)  # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_cov = self.conv_du(cov_mat_sum)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov * x

class IIA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(IIA, self).__init__()
        # self.Capattention = nn.Sequential(
        #     nn.Conv2d(channel, channel*2, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(channel*2, channel, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(channel, 1, 1, stride=1, padding=0, bias=False),
        # )
        self.lm = nn.Conv2d(channel,channel,1,1,0)
        self.Capattention = nn.Sequential(
            nn.Conv2d(channel,  channel//2, 3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channel//2, channel//4, 3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel//4, channel, 1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
    def forward(self, prevousinput,x):
        b, c, w, h = x.size()
        ii_feature = self.lm(x) - self.lm(prevousinput)

        atten = self.Capattention(ii_feature)


        return x*atten



## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True),
                 res_scale=1, dilation=2):
        super(RB, self).__init__()

        self.n_feat = n_feat
        self.gamma1 = 1.0
        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )

        self.res_scale = res_scale

    def forward(self, x):
        b,c,h,w = x.size()
        y1 = self.conv_first(x)
        y = y1 + x

        return y


## Local-source Residual Attention Group (LSRARG)
class RAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(RAG, self).__init__()
        ##
        # body = [RB(conv, n_feat, kernel_size, reduction, \
        #                               bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in
        #                            range(n_resblocks)]
        # self.rbs = nn.Sequential(*body)

        self.rbs = nn.ModuleList([RB(conv, n_feat, kernel_size, reduction, \
                                      bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in
                                   range(n_resblocks)])

        self.iia = (IIA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        previous = x

        for i, l in enumerate(self.rbs):
            # x = l(x) + self.gamma*residual
            x = l(x)

        x = self.iia(previous,x)

        x = self.conv_last(x)
        x = x + previous

        return x


## Information-increment
class IIAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(IIAN, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(inplace=True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.gamma = nn.Parameter(torch.zeros(1))

        self.n_resgroups = n_resgroups
        body = [RAG(conv, n_feats, kernel_size, reduction, \
                                       act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in
                                 range(n_resgroups)]
        self.body = nn.Sequential(*body)

        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)

        self.tail = nn.Sequential(*modules_tail)

        self.iia = IIA(channel=n_feats)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)


        res = self.body(x)

        x = x + res


        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))




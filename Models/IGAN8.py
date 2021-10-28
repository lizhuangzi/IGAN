from Models import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.WaveletTransform import WaveletTransform
#
def make_model(args, parent=False):
    return IGAN(args)



class IGA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(IGA, self).__init__()
        # self.Capattention = nn.Sequential(
        #     nn.Conv2d(channel, channel*2, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(channel*2, channel, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(channel, 1, 1, stride=1, padding=0, bias=False),
        # )
        self.Capattention = nn.Sequential(
            nn.Conv2d(channel*4,  channel, 1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel, channel//4, 3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel//4, channel, 3, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        self.wavedec = WaveletTransform(scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True,
                                        channel=channel)

    def forward(self, prevousinput,x):
        b, c, w, h = x.size()
        ii_feature = x - prevousinput
        ii_feature_4 = self.wavedec(ii_feature)
        atten = self.Capattention(ii_feature_4)


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

        self.iga = (IGA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        previous = x

        for i, l in enumerate(self.rbs):
            # x = l(x) + self.gamma*residual
            x = l(x)

        x = self.iga(previous,x)

        x = self.conv_last(x)
        x = x + previous

        return x


class IGAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(IGAN, self).__init__()
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




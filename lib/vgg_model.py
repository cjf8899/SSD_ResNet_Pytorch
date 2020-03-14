


#  VGG 16 - model 
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init



class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)
        

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x


def VGG():
    base_vgg = models.vgg16().features
    base_vgg[16].ceil_mode = True
    vgg = []
    for i in range(30):
        vgg.append(base_vgg[i])
        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    vgg += [pool5, conv6,nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return vgg
    
    
def Extra():
    layers = []
    conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
    conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
    conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
    conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
    conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
    conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)

    layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2]

    return layers


def Feature_extractor(vgg, extral, bboxes, num_classes):
    
    loc_layers = []
    conf_layers = []
    
    vgg_useful = [21, 33]
    
    for k, v in enumerate(vgg_useful):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 bboxes[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        bboxes[k] * num_classes, kernel_size=3, padding=1)]
    
    for k, v in enumerate(extral[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                  * num_classes, kernel_size=3, padding=1)]
        
    
    
    return loc_layers, conf_layers 




class VGG_SSD(nn.Module):

    def __init__(self, num_classes, bboxes):
        super(VGG_SSD, self).__init__()

        self.num_classes = num_classes
        self.bboxes = bboxes     

        self.vgg_list = VGG()
        self.extra_list = Extra()

        self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.vgg_list, self.extra_list, self.bboxes, self.num_classes)

        self.L2Norm = L2Norm(512, 20)


        self.vgg = nn.ModuleList(self.vgg_list)
        self.extras = nn.ModuleList(self.extra_list)
        self.loc = nn.ModuleList(self.loc_layers_list)
        self.conf = nn.ModuleList(self.conf_layers_list)
        
        
        
#    xavier initialization
#         layers = [self.extras, self.loc, self.conf]
# #         print(self.vgg)
#         for i in layers:
#             for m in i.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.xavier_uniform_(m.weight)
#                     nn.init.zeros_(m.bias)



    def forward(self, x):

        
        source = []
        loc = []
        conf = []


        vgg_source = [22, 34]
        for i, v in enumerate(self.vgg):
            x = v(x)

            if i in vgg_source:
                if i == 22:
                    s = self.L2Norm(x)
                else:
                    s = x
                source.append(s)

        for i, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                source.append(x)


        for s, l, c in zip(source, self.loc, self.conf):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)



        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        return loc, conf



if __name__ == '__main__':

    x = torch.randn(1, 3, 300, 300) 
    ssd = VGG_SSD(21, [4,6,6,6,4,4])
#     print(ssd)
    y = ssd(x)
    print(y[0].shape, y[1].shape)
    
    
    
    
   
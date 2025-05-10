import torch
import torch.nn as nn

from core.nn import _ConvBNReLU

from ..utils.misc import NestedTensor

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class UpsamplingLayer(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, upscale = 2):
        super().__init__()
        self.dim = dim
        
        self.bilinear_upsample = nn.UpsamplingBilinear2d(scale_factor=upscale)
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=3, padding=1)
        self.att = CBAM(dim)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        ''' x: B C H W '''
        x = self.bilinear_upsample(x)
        x = self.att(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

class YOLOWDETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        self.up_test1=UpsamplingLayer(2048, 192)
        self.up_test2=UpsamplingLayer(192+1024, 128)
        self.up_test3=UpsamplingLayer(128+512, 64)
        self.final_test=UpsamplingLayer(64, 19, upscale=4)

        self.up = UpsamplingLayer(512, 512, upscale=4)
        self.final = UpsamplingLayer(512+96, 19, upscale=8)

        self.final_seg = nn.Sequential(
            _ConvBNReLU(512+96, 256, 3, padding=1, norm_layer=nn.BatchNorm2d),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=nn.BatchNorm2d),
            nn.Dropout(0.1),
            nn.Conv2d(256,19, 1))

    def forward(self, samples: NestedTensor):
        class_texts=[
                     "Road",
                     "Sidewalk",
                     "Building",
                     "Wall",
                     "Fence",
                     "Pole",
                     "Traffic Light",
                     "Traffic Sign",
                     "Vegetation",
                     "Terrain",
                     "Sky",
                     "People",
                     "Rider",
                     "Car",
                     "Truck",
                     "Bus",
                     "Train",
                     "Motorcycle",
                     "Bicycle"
                     ]
        
        self.detr.reparameterize([class_texts]*(samples.shape[0]))
        features,_ = self.detr.extract_feat(samples,None)

        layer1 = features[0]
        layer2 = features[1]
        layer3 = features[2]
        x = layer3
        x = self.up_test1(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.up_test2(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.up_test3(x)
        masks = self.final_test(x)
        return masks

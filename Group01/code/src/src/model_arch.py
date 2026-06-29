import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import collections
import numpy as np

from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch


# 定义双线性插值上采样模块（替代 ConvTranspose2d）
class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, bias=False):
        super(UpsampleConv, self).__init__()
        self.scale_factor = scale_factor
        # 双线性插值上采样
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # 随后接卷积层进行特征变换
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UnetVggMultihead(nn.Module):
    def __init__(self, load_weights=False, kwargs=None):
        super(UnetVggMultihead, self).__init__()

        # 预定义参数列表；若外部通过 kwargs 传入同名参数，则会覆盖这里的默认值
        args = {'conv_init': 'he', 'block_size': 3, 'pool_size': 2
            , 'dropout_prob': 0, 'initial_pad': 0, 'n_classes': 1, 'n_channels': 3, 'n_heads': 2, 'head_classes': [1, 1]
                }

        if (not (kwargs is None)):
            args.update(kwargs)

        # 读取并保存外部传入的结构参数
        self.n_channels = int(args['n_channels'])
        self.n_classes = int(args['n_classes'])
        self.conv_init = str(args['conv_init']).lower()
        self.n_heads = int(args['n_heads'])
        self.head_classes = np.array(args['head_classes']).astype(int)

        self.block_size = int(args['block_size'])
        self.pool_size = int(args['pool_size'])
        self.dropout_prob = float(args['dropout_prob'])
        self.initial_pad = int(args['initial_pad'])

        # U-Net 收缩路径（编码器）
        self.encoder = nn.Sequential()
        layer_index = 0
        layer = nn.Sequential()
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_0',
                         nn.Conv2d(self.n_channels, 64, kernel_size=self.block_size, padding=self.initial_pad))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_1',
                         nn.Conv2d(64, 64, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_' + str(layer_index), layer)

        layer_index = 1
        layer = nn.Sequential()
        layer.add_module('encoder_maxpool_l_' + str(layer_index),
                         nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        layer.add_module('encoder_dropout_l_' + str(layer_index), nn.Dropout(p=self.dropout_prob))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_0',
                         nn.Conv2d(64, 128, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_1',
                         nn.Conv2d(128, 128, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_' + str(layer_index), layer)

        layer_index = 2
        layer = nn.Sequential()
        layer.add_module('encoder_maxpool_l_' + str(layer_index),
                         nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        layer.add_module('encoder_dropout_l_' + str(layer_index), nn.Dropout(p=self.dropout_prob))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_0',
                         nn.Conv2d(128, 256, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_1',
                         nn.Conv2d(256, 256, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_2',
                         nn.Conv2d(256, 256, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_' + str(layer_index), layer)

        layer_index = 3
        layer = nn.Sequential()
        layer.add_module('encoder_maxpool_l_' + str(layer_index),
                         nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        layer.add_module('encoder_dropout_l_' + str(layer_index), nn.Dropout(p=self.dropout_prob))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_0',
                         nn.Conv2d(256, 512, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_1',
                         nn.Conv2d(512, 512, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_' + str(layer_index) + '_2',
                         nn.Conv2d(512, 512, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_' + str(layer_index) + '_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_' + str(layer_index), layer)

        # 瓶颈层
        self.bottleneck = nn.Sequential()
        self.bottleneck.add_module('bottleneck_maxpool', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        self.bottleneck.add_module('bottleneck_dropout', nn.Dropout(p=self.dropout_prob))
        self.bottleneck.add_module('bottleneck_conv_0', nn.Conv2d(512, 512, kernel_size=self.block_size))
        self.bottleneck.add_module('bottleneck_relu_0', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv_1', nn.Conv2d(512, 512, kernel_size=self.block_size))
        self.bottleneck.add_module('bottleneck_relu_1', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv_2', nn.Conv2d(512, 512, kernel_size=self.block_size))
        self.bottleneck.add_module('bottleneck_relu_2', nn.ReLU(inplace=True))

        # U-Net 扩张路径（解码器）
        self.decoder = nn.Sequential()
        layer_index = 3
        layer = nn.Sequential()
        layer.add_module('decoder_upsample_l_' + str(layer_index),
                         UpsampleConv(512, 512, scale_factor=self.pool_size, kernel_size=self.block_size))
        layer.add_module('decoder_conv_l_s_' + str(layer_index) + '_0', nn.Conv2d(1024, 512, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_' + str(layer_index) + '_1', nn.Conv2d(512, 512, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(True))
        self.decoder.add_module('decoder_l_' + str(layer_index), layer)

        layer_index = 2
        layer = nn.Sequential()
        layer.add_module('decoder_upsample_l_' + str(layer_index),
                         UpsampleConv(512, 256, scale_factor=self.pool_size, kernel_size=self.block_size))
        layer.add_module('decoder_conv_l_s_' + str(layer_index) + '_0', nn.Conv2d(512, 256, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_' + str(layer_index) + '_1', nn.Conv2d(256, 256, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(True))
        self.decoder.add_module('decoder_l_' + str(layer_index), layer)

        layer_index = 1
        layer = nn.Sequential()
        layer.add_module('decoder_upsample_l_' + str(layer_index),
                         UpsampleConv(256, 128, scale_factor=self.pool_size, kernel_size=self.block_size))
        layer.add_module('decoder_conv_l_s_' + str(layer_index) + '_0', nn.Conv2d(256, 128, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_' + str(layer_index) + '_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_' + str(layer_index) + '_1', nn.Conv2d(128, 128, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_' + str(layer_index) + '_1', nn.ReLU(True))
        self.decoder.add_module('decoder_l_' + str(layer_index), layer)

        layer_index = 0
        layer = nn.Sequential()
        layer.add_module('decoder_upsample_l_' + str(layer_index),
                         UpsampleConv(128, 96, scale_factor=self.pool_size, kernel_size=self.block_size))
        self.decoder.add_module('decoder_l_' + str(layer_index), layer)

        # 多输出头
        self.final_layers_lst = nn.ModuleList()
        for i in range(self.n_heads):
            block = nn.Sequential()
            feat_subblock = nn.Sequential()
            pred_subblock = nn.Sequential()
            feat_subblock.add_module('final_block_' + str(i) + '_conv3_0', nn.Conv2d(96, 64, kernel_size=self.block_size))
            feat_subblock.add_module('final_block_' + str(i) + '_relu_0', nn.ReLU(inplace=True))
            feat_subblock.add_module('final_block_' + str(i) + '_conv3_1', nn.Conv2d(64, 64, kernel_size=self.block_size))
            feat_subblock.add_module('final_block_' + str(i) + '_relu_1', nn.ReLU(True))
            pred_subblock.add_module('final_block_' + str(i) + '_conv1_2', nn.Conv2d(64, self.head_classes[i], kernel_size=1))
            block.add_module('final_block_' + str(i) + 'feat', feat_subblock)
            block.add_module('final_block_' + str(i) + 'pred', pred_subblock)
            self.final_layers_lst.append(block)

        self._initialize_weights()
        self.zero_grad()

    def forward(self, x, feat_indx_list=[], feat_as_dict=False):
        feat = None
        feat_dict = {}
        feat_indx = 0
        encoder_out = []

        # 编码器
        for l in self.encoder:
            x = l(x)
            encoder_out.append(x)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器
        j = len(self.decoder)
        for l in self.decoder:
            x = l[0](x)
            j -= 1
            corresponding_layer_indx = j

            if j > 0:
                cropped = CNNArchUtilsPyTorch.crop_a_to_b(encoder_out[corresponding_layer_indx], x)
                x = torch.cat((cropped, x), 1)
            for i in range(1, len(l)):
                x = l[i](x)

        # 检查是否需要把解码器最终特征作为附加输出返回
        if feat_indx in feat_indx_list:
            if feat_as_dict:
                feat_dict[feat_indx] = x.detach().cpu().numpy()
            else:
                feat = x.detach().cpu().numpy()

        # 输出头
        c = []
        f = None
        for layer in self.final_layers_lst:
            feat_indx += 1
            f1 = layer[0](x)
            c.append(layer[1](f1))

            if f is None:
                f = f1
            else:
                f = torch.cat((f1, f), 1)

            if feat_indx in feat_indx_list:
                if feat_as_dict:
                    feat_dict[feat_indx] = f1.detach().cpu().numpy()
                else:
                    if feat is None:
                        feat = f1.detach().cpu().numpy()
                    else:
                        feat = np.concatenate((feat, f1.detach().cpu().numpy()), axis=1)

        if len(feat_indx_list) == 0:
            return c

        if feat_as_dict:
            return c, feat_dict
        return c, feat

    def _initialize_weights(self):
        # 初始化编码器卷积层参数
        for l in self.encoder:
            for layer in l:
                if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                    if self.conv_init == 'normal':
                        torch.nn.init.normal_(layer.weight)
                    elif self.conv_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(layer.weight)
                    elif self.conv_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(layer.weight, gain=10)
                    elif self.conv_init == 'he':
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # 初始化瓶颈层卷积参数
        for layer in self.bottleneck:
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                if self.conv_init == 'normal':
                    torch.nn.init.normal_(layer.weight)
                elif self.conv_init == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(layer.weight)
                elif self.conv_init == 'xavier_normal':
                    torch.nn.init.xavier_normal_(layer.weight, gain=10)
                elif self.conv_init == 'he':
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # 初始化解码器卷积参数
        for l in self.decoder:
            for layer in l:
                if isinstance(layer, UpsampleConv):
                    if self.conv_init == 'normal':
                        torch.nn.init.normal_(layer.conv.weight)
                    elif self.conv_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(layer.conv.weight)
                    elif self.conv_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(layer.conv.weight, gain=10)
                    elif self.conv_init == 'he':
                        torch.nn.init.kaiming_normal_(layer.conv.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(layer, 'bn') and layer.bn is not None:
                        torch.nn.init.constant_(layer.bn.weight, 1)
                        torch.nn.init.constant_(layer.bn.bias, 0)
                elif isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                    if self.conv_init == 'normal':
                        torch.nn.init.normal_(layer.weight)
                    elif self.conv_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(layer.weight)
                    elif self.conv_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(layer.weight, gain=10)
                    elif self.conv_init == 'he':
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # 初始化各个输出头中的卷积参数
        for layer in self.final_layers_lst:
            for sub_layer in layer:
                for sub_sub_layer in sub_layer:
                    if isinstance(sub_sub_layer, nn.ConvTranspose2d) or isinstance(sub_sub_layer, nn.Conv2d):
                        if self.conv_init == 'normal':
                            torch.nn.init.normal_(sub_sub_layer.weight)
                        elif self.conv_init == 'xavier_uniform':
                            torch.nn.init.xavier_uniform_(sub_sub_layer.weight)
                        elif self.conv_init == 'xavier_normal':
                            torch.nn.init.xavier_normal_(sub_sub_layer.weight, gain=10)
                        elif self.conv_init == 'he':
                            torch.nn.init.kaiming_normal_(sub_sub_layer.weight, mode='fan_out', nonlinearity='relu')

        # 用预训练 VGG-16 参数初始化编码器和瓶颈层
        try:
            vgg_model = models.vgg16(pretrained=True)
            fsd = collections.OrderedDict()
            i = 0
            # 加载编码器权重
            for m in self.encoder.state_dict().items():
                temp_key = m[0]
                fsd[temp_key] = list(vgg_model.state_dict().items())[i][1]
                i += 1
            self.encoder.load_state_dict(fsd)

            # 加载瓶颈层权重
            fsd = collections.OrderedDict()
            for m in self.bottleneck.state_dict().items():
                temp_key = m[0]
                fsd[temp_key] = list(vgg_model.state_dict().items())[i][1]
                i += 1
            self.bottleneck.load_state_dict(fsd)
        except Exception as e:
            print(f"Warning: Could not load VGG16 pretrained weights: {e}")

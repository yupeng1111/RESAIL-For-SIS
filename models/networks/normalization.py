"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import re
# import sys
# sys.path.append('/home/work/workplace/SPADE')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, sean=False, **unused_args):
        super().__init__()
        self.sean = sean
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, params=None):
        if self.sean:
            inputmap = segmap

            actv = self.mlp_shared(inputmap)
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)

            return gamma, beta

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class No_norm(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, **unused_args):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_gamma = nn.Conv2d(norm_nc + label_nc + 3, norm_nc, kernel_size=3, padding=1)


    def forward(self, x, segmap, params=None):
        retrieval = params["retrieval_image"]
        retrieval = F.interpolate(retrieval, size=x.shape[2:], mode='bilinear')
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        out = torch.cat([x, segmap, retrieval], dim=1)
        out = self.mlp_gamma(out)
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(out)

        return normalized

class SPADE_CAT(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, **unused_args):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc + 3, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, params=None):
        retrieval = params["retrieval_image"]
        retrieval = F.interpolate(retrieval, size=x.shape[2:], mode='bilinear')
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(torch.cat([segmap, retrieval], dim=1))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class ACE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        ACE_Name = None
        status = 'train'
        spade_params = None
        use_rgb = True

        self.ACE_Name = ACE_Name
        self.status = status
        self.save_npy = True
        self.Spade = SPADE(config_text, norm_nc, label_nc, sean=True)
        self.use_rgb = use_rgb
        self.style_length = 128
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)


        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.


        if self.use_rgb:
            self.create_gamma_beta_fc_layers()

            self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
            self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)




    def forward(self, x, segmap, params):
        style_codes = params['style_codes']
        obj_dic = None

        # Part 1. generate parameter-free normalized activations

        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if self.use_rgb:
            [b_size, f_size, h_size, w_size] = normalized.shape
            middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=normalized.device)

            if self.status == 'UI_mode':
                ############## hard coding

                for i in range(1):
                    for j in range(segmap.shape[1]):

                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:
                            if obj_dic is None:
                                print('wrong even it is the first input')
                            else:
                                style_code_tmp = obj_dic[str(j)]['ACE']

                                middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_code_tmp))
                                component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,component_mask_area)

                                middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)

            else:

                for i in range(b_size):
                    for j in range(segmap.shape[1]):
                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:


                            middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][j]))
                            component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length, component_mask_area)

                            middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)





            gamma_avg = self.conv_gamma(middle_avg)
            beta_avg = self.conv_beta(middle_avg)


            gamma_spade, beta_spade = self.Spade(None, segmap)

            gamma_alpha = F.sigmoid(self.blending_gamma)
            beta_alpha = F.sigmoid(self.blending_beta)

            gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
            beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
            out = normalized * (1 + gamma_final) + beta_final
        else:
            gamma_spade, beta_spade = self.Spade(segmap)
            gamma_final = gamma_spade
            beta_final = beta_spade
            out = normalized * (1 + gamma_final) + beta_final

        return out





    def create_gamma_beta_fc_layers(self):


        ###################  These codes should be replaced with torch.nn.ModuleList

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        self.fc_mu8 = nn.Linear(style_length, style_length)
        self.fc_mu9 = nn.Linear(style_length, style_length)
        self.fc_mu10 = nn.Linear(style_length, style_length)
        self.fc_mu11 = nn.Linear(style_length, style_length)
        self.fc_mu12 = nn.Linear(style_length, style_length)
        self.fc_mu13 = nn.Linear(style_length, style_length)
        self.fc_mu14 = nn.Linear(style_length, style_length)
        self.fc_mu15 = nn.Linear(style_length, style_length)
        self.fc_mu16 = nn.Linear(style_length, style_length)
        self.fc_mu17 = nn.Linear(style_length, style_length)
        self.fc_mu18 = nn.Linear(style_length, style_length)
        self.fc_mu19 = nn.Linear(style_length, style_length)
        self.fc_mu20 = nn.Linear(style_length, style_length)
        self.fc_mu21 = nn.Linear(style_length, style_length)
        self.fc_mu22 = nn.Linear(style_length, style_length)
        self.fc_mu23 = nn.Linear(style_length, style_length)
        self.fc_mu24 = nn.Linear(style_length, style_length)
        self.fc_mu25 = nn.Linear(style_length, style_length)
        self.fc_mu26 = nn.Linear(style_length, style_length)
        self.fc_mu27 = nn.Linear(style_length, style_length)
        self.fc_mu28 = nn.Linear(style_length, style_length)
        self.fc_mu29 = nn.Linear(style_length, style_length)
        self.fc_mu30 = nn.Linear(style_length, style_length)
        self.fc_mu31 = nn.Linear(style_length, style_length)
        self.fc_mu32 = nn.Linear(style_length, style_length)
        self.fc_mu33 = nn.Linear(style_length, style_length)
        self.fc_mu34 = nn.Linear(style_length, style_length)

# for sean
class Zencoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):

        codes = self.model(input)

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        # print(segmap.shape)
        # print(codes.shape)


        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)


        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)

        return codes_vector



class SIN(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        self.style_length = 128
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        self.conv1 = torch.nn.Conv2d(3, self.style_length, 3, 1, padding=1)
        self.adaIn1 = AdaIN(norm_nc * 2)
        self.relu1 = nn.ReLU()

        self.conv2 = torch.nn.Conv2d(self.style_length, self.style_length, 3, 1, padding=1)
        self.adaIn2 = AdaIN(norm_nc * 2)
        self.relu2 = nn.ReLU()

        self.conv3 = torch.nn.Conv2d(self.style_length, self.style_length, 3, 1, padding=1)

        self.conv_s = torch.nn.Conv2d(label_nc, self.style_length * 2, 3, 2)

        self.pool_s = torch.nn.AdaptiveAvgPool2d(1)

        self.conv_s2 = torch.nn.Conv2d(self.style_length * 2, self.style_length * 2, 1, 1)

        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)

        n_hidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, n_hidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(n_hidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(n_hidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, seg_map, params):
        retrieval = params["retrieval_image"]
        normalized = self.param_free_norm(x)

        retrieval = F.interpolate(retrieval, size=x.shape[2:], mode='bilinear')
        seg_map = F.interpolate(seg_map, size=x.shape[2:], mode='nearest')

        f_s_1 = self.conv_s(seg_map)
        c1 = self.pool_s(f_s_1)
        c2 = self.conv_s2(c1)

        f1 = self.conv1(retrieval)

        f1 = self.adaIn1(f1, c1[:, : self.style_length, ...], c1[:, self.style_length:, ...])
        f2 = self.relu1(f1)

        f2 = self.conv2(f2)
        f2 = self.adaIn2(f2, c2[:, : self.style_length, ...], c2[:, self.style_length:, ...])
        f2 = self.relu2(f2)
        middle_avg = self.conv3(f2)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)

        active = self.mlp_shared(seg_map)

        gamma_spade, beta_spade = self.mlp_gamma(active), self.mlp_beta(active)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final

        return out


class AdaIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

    def forward(self, x, alpha, gamma):
        assert x.shape[:2] == alpha.shape[:2] == gamma.shape[:2]
        norm = self.instance_norm(x)
        return alpha * norm + gamma


class StyleSPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 256

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.adaIn_params_projection1 = nn.Sequential(
            nn.Linear(512, nhidden * 2)
        )

        # self.adaIn_params_projection2 = nn.Sequential(
        #     nn.Linear(nhidden * 2, nhidden * 2)
        # )
        self.adaIn1 = AdaIN(nhidden)
        # self.adaIn2 = AdaIN(nhidden)

        self.conv_s1 = nn.Sequential(
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.conv_s2 = nn.Sequential(
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, params=None):
        z = params['z']

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        conv1 = self.mlp_shared(segmap)
        z1 = self.adaIn_params_projection1(z)

        s1 = self.adaIn1(conv1, z1[:, : 256].unsqueeze(-1).unsqueeze(-1), z1[:, 256:].unsqueeze(-1).unsqueeze(-1))

        s2 = self.conv_s1(s1)


        actv = self.conv_s2(s2)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class LightSPADE(nn.Module):

    def __init__(self, norm_nc, label_nc):
        super().__init__()

        ks = 3

        self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )


        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):


        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SIN_SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        self.style_length = 128
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)


        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)

        n_hidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, n_hidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(n_hidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(n_hidden, norm_nc, kernel_size=ks, padding=pw)

        self.spade1 = LightSPADE(3, label_nc)
        self.spade2 = LightSPADE(3, label_nc)
        self.conv3 = nn.Conv2d(3, n_hidden, kernel_size=ks, padding=pw)

    def forward(self, x, seg_map, params):

        retrieval = params["retrieval_image"]
        normalized = self.param_free_norm(x)

        retrieval = F.interpolate(retrieval, size=x.shape[2:], mode='bilinear')

        seg_map = F.interpolate(seg_map, size=x.shape[2:], mode='nearest')

        spade1 = self.spade1(retrieval, seg_map)
        spade1 = self.relu1(spade1)
        spade2 = self.spade2(spade1, seg_map)
        spade2 = self.relu2(spade2)

        middle_avg = self.conv3(spade2)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)

        active = self.mlp_shared(seg_map)

        gamma_spade, beta_spade = self.mlp_gamma(active), self.mlp_beta(active)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final

        return out
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import models.networks as networks
import util.util as util


from segmentation.segnet import SegNet


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        parser.add_argument('--segloss', action="store_true")
        parser.add_argument('--no_retrieval_loss', action="store_true")
        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        if opt.segloss and self.opt.isTrain:
            self.segnet = SegNet(opt).cuda()
            if self.opt.dataset_mode == "ade20k":
                print('logging from ./segmentation/ade20k/80_net_Seg.pth')
                self.segnet.load_state_dict(torch.load('./segmentation/ade20k/80_net_Seg.pth'))
            if self.opt.dataset_mode == "ade20koutdoor":
                print('logging from ./segmentation/ade20koutdoor/80_net_Seg.pth')
                self.segnet.load_state_dict(torch.load('./segmentation/ade20k/80_net_Seg.pth'))
            elif self.opt.dataset_mode == "cityscapes":
                self.segnet.load_state_dict(torch.load('./segmentation/cityscapes/80_net_Seg.pth'))
            elif self.opt.dataset_mode == "coco":
                self.segnet.load_state_dict(torch.load('./segmentation/coco/40_net_Seg.pth'))
            else:
                raise NotImplementedError()

            self.segnet.eval()

            def balanced_NLLloss2d(pred_label, real_label, coff):
                real_label = torch.argmax(real_label, dim=1).long()

                loss = F.cross_entropy(pred_label, real_label, reduction='none')

                l = torch.mean(loss * coff[:, 0, ...])

                return l

            self.criterion_seg_loss = balanced_NLLloss2d

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':

            with torch.no_grad():
                fake_image, _ = self.generate_fake(
                    self.input_semantics,
                    real_image,
                    params={"retrieval_image": self.retrieval_image}
                )

            return fake_image

        elif mode == 'generator_retrieval':
            g_loss, generated = self.compute_retrieval_generator_loss()
            return g_loss, generated
        elif mode == 'discriminator_retrieval':
            d_loss = self.compute_retrieval_discriminator_loss()
            return d_loss

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)


        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        self.retrieval_image = data["retrieval_image"].cuda(non_blocking=True)
        self.modified_image = data['modified_image'].cuda(non_blocking=True)
        self.real_image = data['image'].cuda(non_blocking=True)

        # create one-hot label map
        label_map = data['class_label'].cuda(non_blocking=True).long()
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        self.input_semantics = input_semantics

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
        if self.opt.segloss:

            self.loss_coff = balance_label(self.input_semantics)


        return input_semantics, data['image'].cuda(non_blocking=True)

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        # if not self.opt.no_retrieval:
        if True:
            params = {"retrieval_image": self.modified_image}

            fake_image, KLD_loss = self.generate_fake(
                input_semantics,
                real_image,
                compute_kld_loss=self.opt.use_vae,
                params=params
            )

            if self.opt.use_vae:
                G_losses['KLD'] = KLD_loss

            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

            G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)

            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

            if not self.opt.no_vgg_loss:
                G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                                  * self.opt.lambda_vgg
            if self.opt.segloss:
                pred_seg_map = self.segnet(fake_image)
                # todo change the class number.
                G_losses['GAN_seg_loss_1'] = self.criterion_seg_loss(pred_seg_map, self.input_semantics[:, : 35], self.loss_coff) * 5

        retrieval_loss, fake_image_with_retrieval_segment = self.compute_retrieval_generator_loss()

        return {**G_losses, **retrieval_loss}, fake_image

    def compute_retrieval_generator_loss(self):
        G_losses = {}
        if self.opt.no_retrieval_loss:
            return G_losses, None
        fake_image_with_retrieval_instance, _ = self.generate_fake(
            self.input_semantics,
            self.real_image,
            compute_kld_loss=self.opt.use_vae,
            params={"retrieval_image": self.retrieval_image}
        )


        pred_fake_for_retrieval_instance, pred_real_for_retrieval_instance = self.discriminate(
            self.input_semantics, fake_image_with_retrieval_instance, self.real_image)

        G_losses['GAN_for_retrieval_instance'] = self.criterionGAN(pred_fake_for_retrieval_instance, True,
                                                                   for_discriminator=False)

        if self.opt.segloss:
            pred_seg_map = self.segnet(fake_image_with_retrieval_instance)
            # todo change the class number
            G_losses['GAN_seg_loss_retrieval'] = self.criterion_seg_loss(pred_seg_map, self.input_semantics[:, : 35], self.loss_coff) * 5


        return G_losses, fake_image_with_retrieval_instance

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        # if not self.opt.no_retrieval:
        if True:
            params = {"retrieval_image": self.modified_image}

            with torch.no_grad():
                fake_image, _ = self.generate_fake(
                    input_semantics,
                    real_image,
                    params=params
                )
                fake_image = fake_image.detach()
                fake_image.requires_grad_()

            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        retireval_loss = self.compute_retrieval_discriminator_loss()

        return {**D_losses, **retireval_loss}

    def compute_retrieval_discriminator_loss(self):
        input_semantics = self.input_semantics
        D_losses = {}
        if self.opt.no_retrieval_loss:
            return D_losses
        with torch.no_grad():
            fake_image_for_retrieval_segment, _ = self.generate_fake(
                input_semantics,
                self.real_image,
                params={'retrieval_image': self.retrieval_image}
            )
            fake_image_for_retrieval_segment = fake_image_for_retrieval_segment.detach()
            fake_image_for_retrieval_segment.requires_grad_()

        pred_fake_for_retrieval_instance, pred_real_for_retrieval_instance = self.discriminate(
            input_semantics, fake_image_for_retrieval_segment, self.real_image)

        D_losses['D_Fake_for_retrieval'] = self.criterionGAN(pred_fake_for_retrieval_instance, False,
                                                             for_discriminator=True)
        D_losses['D_real_retrieval'] = self.criterionGAN(pred_real_for_retrieval_instance, True,
                                                         for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False, params=None):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z, params=params)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


# todo balance label
def balance_label(input_semantics):
    '''
    label: bxnxwxh
    '''

    label = input_semantics
    class_occurence = torch.sum(label, dim=(0, 2, 3))
    num_of_classes = (class_occurence > 0).sum()
    coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (
            num_of_classes * label.shape[1])
    integers = torch.argmax(label, dim=1, keepdim=True)
    weight_map = coefficients[integers]

    return weight_map
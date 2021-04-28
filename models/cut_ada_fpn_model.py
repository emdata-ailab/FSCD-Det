import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from models.augment import AugmentPipe
import util.training_stats as training_stats 
from util import dnnlib
from util import misc
import cv2
import os


class CUTADAFPNModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample', 'strided_conv'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # parameter initialize
        self.opt = opt
        self.parameter_initialize(opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.loss_names += ['extend_NCE']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionConsistency = torch.nn.L1Loss()
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Setup augmentation
        self.augment_pipe = None
        self.ada_stats = None
        self.augment_pipe = AugmentPipe()
        for key, value in self.augment_kwargs.items():
            setattr(self.augment_pipe, key, value)
        self.augment_pipe = self.augment_pipe.train().requires_grad_(False).to(self.device)
        self.augment_pipe.p.copy_(torch.as_tensor(opt.ada_initial_p))
        self.ada_stats = training_stats.Collector(regex='Loss/signs/real')

    def parameter_initialize(self, opt):
        self.cur_nimg = 0
        self.batch_idx = 0
        self.batch_size = opt.batch_size

        self.ada_target = opt.ada_target
        aug = opt.aug
        self.ada_interval = opt.ada_interval
        augpipe_specs = {
            'bl':   dict(xflip=1, xint=1),
            'blit':   dict(xflip=1, rotate90=1, xint=1),
            'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise':  dict(noise=1),
            'cutout': dict(cutout=1),
            'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }
        assert aug in augpipe_specs
        self.augment_kwargs =  augpipe_specs[aug]
        


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

        #update state
        self.cur_nimg += self.batch_size
        self.batch_idx += 1

        # Execute ADA heuristic
        if self.batch_idx % self.ada_interval == 0:
            self.ada_stats.update()
            adjust = np.sign(self.ada_stats['Loss/signs/real'] - self.ada_target) * (self.batch_size * self.ada_interval) / (self.dataset_size*(self.opt.n_epochs + self.opt.n_epochs_decay))
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=self.device)))


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # For FPN
        self.real_A_extend = []
        if 'A_extend' in input:
            for i in range(len(input['A_extend'])):
                self.real_A_extend.append(input['A_extend'][i].to(self.device))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.horizontal_flip = np.random.random() < self.opt.hflip and self.opt.isTrain
        self.vertical_flip = np.random.random() < self.opt.vflip and self.opt.isTrain
        if self.horizontal_flip and self.vertical_flip:
            self.real = torch.cat((self.real, torch.flip(self.real[:self.real_A.size(0)].detach(), (2,3)).to(self.device)), dim=0)
        elif self.horizontal_flip:
            self.real = torch.cat((self.real, torch.flip(self.real[:self.real_A.size(0)].detach(), (3,)).to(self.device)), dim=0)
        elif self.vertical_flip:
            self.real = torch.cat((self.real, torch.flip(self.real[:self.real_A.size(0)].detach(), (2,)).to(self.device)), dim=0)

        self.fake = self.netG(self.real)
        # Identical B
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):self.real_A.size(0)+self.real_B.size(0)]
        
        # Horizontal and Vertical Flip
        self.flip_fake = None
        if self.vertical_flip and self.horizontal_flip:
            # Flip back to calculate loss
            self.flip_fake = torch.flip(self.fake[-self.real_A.size(0):], (2,3))
        elif self.vertical_flip:
            self.flip_fake = torch.flip(self.fake[-self.real_A.size(0):], (2,))
        elif self.horizontal_flip:
            self.flip_fake = torch.flip(self.fake[-self.real_A.size(0):], (3,))

        # Different Scales
        self.extend_fakes = []
        for extend_A in self.real_A_extend:
            self.extend_fakes.append(self.netG(extend_A))

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Augment
        if self.augment_pipe is not None:
            fake = self.augment_pipe(fake)
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        
        # Fake flip
        if self.vertical_flip or self.horizontal_flip:
            flip_fake = self.flip_fake.detach()
            pred_flip_fake = self.netD(flip_fake)
            self.loss_D_flip_fake = self.criterionGAN(pred_flip_fake, False).mean()
        
        # Real
        if self.augment_pipe is not None:
            real_B_tmp = self.real_B.detach().requires_grad_(True)
            aug_real = self.augment_pipe(real_B_tmp)
        self.pred_real = self.netD(aug_real)#self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        training_stats.report('Loss/signs/real', self.pred_real.sign())

        # combine loss and calculate gradients
        if not (self.vertical_flip or self.horizontal_flip):
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        else:
            self.loss_D = (self.loss_D_fake + self.loss_D_real + self.loss_D_flip_fake) / 3.0
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, augment fake image
        if self.augment_pipe is not None:
            aug_fake = self.augment_pipe(fake)
        # G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(aug_fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # NCE loss
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
            self.loss_extend_NCE = 0
            for extend_real, extend_fake in zip(self.real_A_extend, self.extend_fakes):
                self.loss_extend_NCE += self.calculate_NCE_loss(extend_real, extend_fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        # Flip Consistency Loss
        flip_fake = self.flip_fake
        if self.vertical_flip or self.horizontal_flip:
            pred_flip_fake = self.netD(flip_fake)
            self.loss_G_flip_GAN = self.criterionGAN(pred_flip_fake, True).mean() * self.opt.lambda_GAN
            self.loss_fake_consistent = self.criterionConsistency(self.flip_fake, self.fake_B)
            if self.opt.lambda_NCE > 0.0:
                self.loss_flip_NCE = self.calculate_NCE_loss(self.real_A, self.flip_fake)
        else:
            self.loss_G_flip_GAN = 0.0
            self.loss_flip_NCE = 0.0
        
        # Sum all the loss
        loss_NCE_both = torch.tensor(0.0, requires_grad=True).to(self.device)
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both += self.loss_NCE_Y
        
        if getattr(self, 'loss_extend_NCE', None) is not None:
            loss_NCE_both += self.loss_extend_NCE

        if (self.vertical_flip or self.horizontal_flip) and self.opt.lambda_NCE > 0.0:
            loss_NCE_both += self.loss_flip_NCE
        
        loss_NCE_both += self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        if self.vertical_flip or self.horizontal_flip:
            self.loss_G += self.loss_G_flip_GAN + self.loss_fake_consistent
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
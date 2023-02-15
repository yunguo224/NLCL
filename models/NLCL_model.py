import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .disnce import DisNCELoss
import time
import datetime


class NLCLModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

        parser.add_argument('--lambda_DisNCE', type=float, default=1.0, help='weight for Dis NCE loss')
        parser.add_argument('--lambda_MSE', type=float, default=1.0, help='weight for MSE loss')
        parser.add_argument('--lambda_L1', type=float, default=0.1, help='weight for MSE loss')

        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--adv_nce_layers', type=str, default='0,3,7,11', help='compute NCE loss on which layers')
        parser.add_argument('--gen_nce_layers', type=str, default='0,2,4,8,12', help='compute NCE loss on which layers')
        parser.add_argument('--netFGen', type=str, default='non_localOne',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netFAdvRain', type=str, default='non_localOne',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netFAdvBack', type=str, default='non_localOne',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches_pos', type=int, default=8, help='number of patches per layer')
        parser.add_argument('--num_patches_neg', type=int, default=128, help='number of patches per layer')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0,serial_batches=False)
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

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'NCE', 'DisNCE', 'MSE', 'L1']
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'DisNCE', 'MSE', 'L1']
        self.visual_names = ['O', 'B', 'pred_B', 'pred_R', 'pred_O']
        if not self.isTrain:
            self.visual_names = ['O','pred_B']
        self.adv_nce_layers = [int(i) for i in self.opt.adv_nce_layers.split(',')]
        self.gen_nce_layers = [int(i) for i in self.opt.gen_nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            # self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['Rain', 'Back', 'GenSample', 'AdvRainSample', 'AdvBackSample', 'D']
            self.pretrained_names = ['Rain', 'Back', 'D']
        else:  # during test time, only load G
            self.model_names = ['Rain', 'Back']
            self.pretrained_names = ['Rain', 'Back']

        self.netRain = networks.Generator(inchannel=3,outchannel=3).to(self.device)
        self.netBack = networks.Generator(inchannel=3,outchannel=3).to(self.device)


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netGenSample = networks.define_F(opt.input_nc, opt.netFGen, opt.normG, not opt.no_dropout,
                                                  opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netAdvRainSample = networks.define_F(opt.input_nc, opt.netFAdvRain, opt.normG, not opt.no_dropout,
                                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netAdvBackSample = networks.define_F(opt.input_nc, opt.netFAdvBack, opt.normG, not opt.no_dropout,
                                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionNCE = []

            self.criterDisNCE = DisNCELoss(self.opt).to(self.device)

            for nce_layer in self.gen_nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_Rain = torch.optim.Adam(self.netRain.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_Back = torch.optim.Adam(self.netBack.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_Rain)
            self.optimizers.append(self.optimizer_Back)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.O.size(0) // max(len(self.opt.gpu_ids), 1)
        self.O = self.O[:bs_per_gpu]
        self.B = self.B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        print('data_dependent_initialize forward success')
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_GenSample = torch.optim.Adam(self.netGenSample.parameters(), lr=self.opt.lr,betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_GenSample)

            self.optimizer_AdvRainSample = torch.optim.Adam(self.netAdvRainSample.parameters(), lr=self.opt.lr,betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_AdvBackSample = torch.optim.Adam(self.netAdvRainSample.parameters(), lr=self.opt.lr,betas=(self.opt.beta1, self.opt.beta2))

            self.optimizers.append(self.optimizer_AdvRainSample)
            self.optimizers.append(self.optimizer_AdvBackSample)

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
        self.optimizer_Rain.zero_grad()
        self.optimizer_Back.zero_grad()
        # if self.opt.netF == 'mlp_sample':
        self.optimizer_AdvRainSample.zero_grad()
        self.optimizer_AdvBackSample.zero_grad()
        if self.opt.lambda_NCE != 0:
            self.optimizer_GenSample.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_Rain.step()
        self.optimizer_Back.step()
        if self.opt.lambda_NCE != 0:
            self.optimizer_GenSample.zero_grad()
        self.optimizer_AdvRainSample.step()
        self.optimizer_AdvBackSample.step()

    def set_input(self, input):
        import os
        AtoB = self.opt.direction == 'AtoB'
        self.O = input['A' if AtoB else 'B'].to(self.device)
        self.B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.img_name = input['A_paths'][0]
        self.img_name = os.path.split(self.img_name)
        self.img_name = self.img_name[-1]

    def forward(self):
        self.image_to_transfer = torch.cat((self.O, self.B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.O
        # self.image_to_transfer = self.O
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.image_to_transfer = torch.flip(self.image_to_transfer, [3])
        # torch.cuda.synchronize()
        start_time = datetime.datetime.now()
        (h,w) = self.O.shape[2:]
        self.pred_R = self.netRain(self.O)
        self.image_transferred = self.netBack(self.image_to_transfer)
        # torch.cuda.synchronize()
        end_time = datetime.datetime.now()
        during_time = end_time - start_time
        # print("img size:(%d,%d)test time:%f"%(h,w,during_time))
        print(during_time)
        self.pred_B = self.image_transferred[:self.O.size(0)]
        self.pred_O = self.pred_R + self.pred_B
        if self.opt.nce_idt:
            self.idt_B = self.image_transferred[self.O.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.pred_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.pred_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_MutualNCE_loss(self.O, self.pred_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_MutualNCE_loss(self.B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_DisNCE = self.calculate_DisNCE_loss(self.pred_B, self.pred_R, self.O) * self.opt.lambda_DisNCE

        self.loss_L1  = self.criterionL1(self.O, self.pred_B) * self.opt.lambda_L1
        self.loss_MSE = self.criterionMSE(self.pred_O, self.O) * self.opt.lambda_MSE

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_DisNCE + self.loss_MSE + self.loss_L1
        return self.loss_G

    # mutual NCE(location contrastive)
    def calculate_MutualNCE_loss(self, src, tgt):
        n_layers = len(self.gen_nce_layers)
        feat_b = self.netBack(tgt, self.gen_nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_b = [torch.flip(fq, [3]) for fq in feat_b]

        feat_o = self.netBack(src, self.gen_nce_layers, encode_only=True)
        feat_o_pool, sample_ids = self.netGenSample(feat_o, self.opt.num_patches, None)
        feat_b_pool, _ = self.netGenSample(feat_b, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_b, f_o, crit, nce_layer in zip(feat_b_pool, feat_o_pool, self.criterionNCE, self.gen_nce_layers):
            loss = crit(f_b, f_o) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # disentangle NCE(layer contrastive)
    def calculate_DisNCE_loss(self, pred_B, pred_R, O):

        # timestrap1 = time.time()
        feat_pred_B = self.netD(pred_B, self.adv_nce_layers, encode_only=True)
        feat_pred_R = self.netD(pred_R, self.adv_nce_layers, encode_only=True)
        # timestrap2 = time.time()
        # print('time consume-extract feature:',timestrap2-timestrap1)

        feat_B_pool, _ = self.netAdvBackSample(feat_pred_B, self.opt.num_patches_pos, None)
        feat_R_pool, _ = self.netAdvRainSample(feat_pred_R, self.opt.num_patches_neg, None)
        # feat_B_pool, ids = self.netAdvSample(feat_pred_B, numpatches, None)
        # feat_R_pool, _ = self.netAdvSample(feat_pred_R, numpatches, ids)
        # timestrap1 = time.time()
        # print('time consume-sampling:', timestrap1 - timestrap2)

        total_dis_loss = 0.0
        for f_b, f_r, nce_layer in zip(feat_B_pool, feat_R_pool, self.adv_nce_layers):
            loss = self.criterDisNCE(f_b, f_r)
            total_dis_loss += loss.mean()
        # timestrap2 = time.time()
        # print('time consume-cal loss:', timestrap2- timestrap1)

        return total_dis_loss / len(self.adv_nce_layers)

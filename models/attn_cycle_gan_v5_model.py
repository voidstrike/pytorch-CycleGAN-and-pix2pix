import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from data import aux_dataset
from . import networks


class AttnCycleGANV5Model(BaseModel):
    # Spatial Attention Enhanced Img2Img Translation model
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # Following the same pattern in original CycGAN model
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--mask_size', type=int, default=128,
                            help='The resolution of the mask, should be same with attention map')

        # Configuration for TAM model, if the input is 256*256, s1=64, s2=32 | if the input is 128*128, s1=32, s2=16
        parser.add_argument('--s1', type=int, default=32,
                            help='The size of the output of attention block, first down-sampling')
        parser.add_argument('--s2', type=int, default=16,
                            help='The size of the output of attention block, second down-sampling')

        # Concatenation method, 'alpha' stands for RGBA, 'rmult' stands for 'RHP' & 'none' means no attention
        parser.add_argument('--concat', type=str, default='rmult', help='Concatenation type, alpha|rmult|none')

        if is_train:
            # Disabled in our framework
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        # Initialize the network structure, parameter groups
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # Intermediate result that will be visualized -- change manually
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # visual_names_A = ['real_A', 'fake_B', 'rec_A', 'vis_A2B']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B', 'vis_B2A']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # Disabled in our framework
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'S_CA', 'S_CB']
        else:  # during test time, load required networks
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'S_CA', 'S_CB']

        # define networks (both Generators and discriminators)
        # Initialize the small attention transformation network
        self.netS_CA = networks.define_C(opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netS_CB = networks.define_C(opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        # Define the structure of the generator based on the concatenation type
        if opt.concat != 'alpha':
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            # Extra Channel
            self.netG_A = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Auxiliary attention holder
        self.aux_data = aux_dataset.AuxAttnDataset(7000, 7000, self.gpu_ids[0], mask_size=32)
        self.zero_attn_holder = torch.zeros((1, 1, opt.mask_size, opt.mask_size), dtype=torch.float32).to(self.device)
        self.ones_attn_holder = torch.ones((1, 1, opt.mask_size, opt.mask_size), dtype=torch.float32).to(self.device)

        self.concat = opt.concat

        # Visualization purpose only
        self.vis_A2B, self.vis_B2A = torch.zeros((1, 1, 256, 256), dtype=torch.float32).to(self.device), \
                                     torch.zeros((1, 1, 256, 256), dtype=torch.float32).to(self.device)

        self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                        opt.mask_size, opt.s1, opt.s2)
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                        opt.mask_size, opt.s1, opt.s2)

        if self.isTrain:
            # Initialize components that will only be used in training phase

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                pass
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),
                                                                self.netS_CA.parameters(), self.netS_CB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.attn_A_index = input['ADX' if AtoB else'BDX']
        self.attn_B_index = input['BDX' if AtoB else'ADX']

        # Extract corresponding attention map
        if AtoB:
            self.attn_A, self.attn_B = self.aux_data.get_attn_map(self.attn_A_index, self.attn_B_index)
        else:
            self.attn_B, self.attn_A = self.aux_data.get_attn_map(self.attn_A_index, self.attn_B_index)

        self.image_paths = [input['A_paths'], input['B_paths']]

    def forward(self, mode=False):
        #  Forward phase for training phase.
        #  Mode = True will execute the out-iteration attention & False will leads to in-iteration attention
        if mode:
            concat_attn_A, concat_attn_B = self.netS_CA(self.attn_A), self.netS_CB(self.attn_B)
        else:
            # Disable the gradient of the Discriminator during in-iteration attention
            self.set_requires_grad([self.netD_A, self.netD_B], False)

        #self.vis_A2B, self.vis_B2A = (concat_attn_A - .5) / .5, (concat_attn_B - .5) / .5

        if self.concat == 'alpha':
            if mode:
                self.fake_B = self.netG_A(torch.cat((self.real_A, concat_attn_A), 1))
                self.rec_A = self.netG_B(torch.cat((self.fake_B, self.ones_attn_holder), 1))
                self.fake_A = self.netG_B(torch.cat((self.real_B, concat_attn_B), 1))
                self.rec_B = self.netG_A(torch.cat((self.fake_A, self.ones_attn_holder), 1))
            else:
                self.fake_B = self.netG_A(torch.cat((self.real_A, self.ones_attn_holder), 1))
                self.rec_A = self.netG_B(torch.cat((self.fake_B, self.ones_attn_holder), 1))
                self.fake_A = self.netG_B(torch.cat((self.real_B, self.ones_attn_holder), 1))
                self.rec_B = self.netG_A(torch.cat((self.fake_A, self.ones_attn_holder), 1))

                _, inner_A2B = self.netD_A(self.fake_B)
                _, inner_B2A = self.netD_B(self.fake_A)
                inner_A2B, inner_B2A = self.netS_CA(inner_A2B.detach()), self.netS_CB(inner_B2A.detach())
                _, inner_A2B_prime = self.netD_A(self.rec_B)
                _, inner_B2A_prime = self.netD_B(self.rec_A)
                inner_A2B_prime, inner_B2A_prime = self.netS_CA(inner_A2B_prime.detach()), \
                                                   self.netS_CB(inner_B2A_prime.detach())

                self.fake_B = self.netG_A(torch.cat((self.real_A, inner_A2B), 1))
                self.rec_A = self.netG_B(torch.cat((self.fake_B, inner_B2A_prime), 1))
                self.fake_A = self.netG_B(torch.cat((self.real_B, inner_B2A), 1))
                self.rec_B = self.netG_A(torch.cat((self.fake_A, inner_A2B_prime), 1))

        elif self.concat == 'rmult':
            if mode:
                self.fake_B = self.netG_A(self.real_A * (1. + concat_attn_A))
                self.rec_A = self.netG_B(self.fake_B * 1.5)
                self.fake_A = self.netG_B(self.real_B * (1. + concat_attn_B))
                self.rec_B = self.netG_A(self.fake_A * 1.5)
            else:
                self.fake_B = self.netG_A(self.real_A * 1.5)
                self.rec_A = self.netG_B(self.fake_B * 1.5)
                self.fake_A = self.netG_B(self.real_B * 1.5)
                self.rec_B = self.netG_A(self.fake_A * 1.5)

                _, inner_A2B = self.netD_A(self.fake_B)
                _, inner_B2A = self.netD_B(self.fake_A)
                inner_A2B, inner_B2A = self.netS_CA(inner_A2B.detach()), self.netS_CB(inner_B2A.detach())
                _, inner_A2B_prime = self.netD_A(self.rec_B)
                _, inner_B2A_prime = self.netD_B(self.rec_A)
                inner_A2B_prime, inner_B2A_prime = self.netS_CA(inner_A2B_prime.detach()), \
                                                   self.netS_CB(inner_B2A_prime.detach())

                self.fake_B = self.netG_A(self.real_A * (1. + inner_A2B))
                self.rec_A = self.netG_B(self.fake_B * (1. + inner_B2A_prime))
                self.fake_A = self.netG_B(self.real_B * (1. + inner_B2A))
                self.rec_B = self.netG_A(self.fake_A * (1. + inner_A2B_prime))

        elif self.concat == 'none':
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        else:
            raise NotImplementedError('Unsupported concatenation operation')

    def forward_test(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Consider More here
        concat_attn_A, concat_attn_B = self.netS_CA(self.attn_A), self.netS_CB(self.attn_B)

        if self.concat == 'alpha':
            self.fake_B = self.netG_A(torch.cat((self.real_A, concat_attn_A), 1))
            self.rec_A = self.netG_B(torch.cat((self.fake_B, self.ones_attn_holder), 1))
            self.fake_A = self.netG_B(torch.cat((self.real_B, concat_attn_B), 1))
            self.rec_B = self.netG_A(torch.cat((self.fake_A, self.ones_attn_holder), 1))
        elif self.concat == 'rmult':
            self.fake_B = self.netG_A(self.real_A * (1. + concat_attn_A))
            self.rec_A = self.netG_B(self.fake_B * 1.5)
            self.fake_A = self.netG_B(self.real_B * (1. + concat_attn_B))
            self.rec_B = self.netG_A(self.fake_A * 1.5)
        elif self.concat == 'none':
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        else:
            raise NotImplementedError('Unsupported concatenation operation')

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, _ = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake, _ = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, mode=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss -- Always be zero in final version
        if lambda_idt > 0:
            if self.concat == 'alpha':
                self.idt_A = self.netG_A(torch.cat((self.real_B, self.ones_attn_holder), 1))
                self.idt_B = self.netG_B(torch.cat((self.real_A, self.ones_attn_holder), 1))
            else:
                self.idt_A = self.netG_A(self.real_B)
                self.idt_B = self.netG_B(self.real_A)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        dis_A_res, self.tmp_attn_A = self.netD_A(self.fake_B)
        dis_B_res, self.tmp_attn_B = self.netD_B(self.fake_A)
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(dis_A_res, True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(dis_B_res, True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.aux_data.update_attn_map(self.attn_A_index, self.tmp_attn_A.detach(), True)
        self.aux_data.update_attn_map(self.attn_B_index, self.tmp_attn_B.detach(), False)

    def optimize_parameters_attn(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.aux_data.update_attn_map(self.attn_A_index, self.tmp_attn_A.detach(), True)
        self.aux_data.update_attn_map(self.attn_B_index, self.tmp_attn_B.detach(), False)

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward_test()
            self.compute_visuals()


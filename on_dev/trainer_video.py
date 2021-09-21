import time
import torch
import datetime
import os
from torch._C import _log_api_usage_once

import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR, MultiStepLR
from torch.autograd import Variable

from models.mocogan.generator import VideoGenerator
from models.mocogan.discriminator import PatchImageDiscriminator
from models.mocogan.discriminator import PatchVideoDiscriminator
from utils.utils import *


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.MODEL.NAME
        self.adv_loss = config.TRAIN.LOSS

        # Model hyper-parameters
        self.imsize = config.MODEL.IMAGE_SIZE
        self.g_num = config.MODEL.G_NUM
        self.z_dim = config.MODEL.Z_DIM
        self.g_chn = config.MODEL.G_CHANNEL
        self.ds_chn = config.MODEL.DS_CHANNEL
        self.dt_chn = config.MODEL.DT_CHANNEL
        self.n_frames = config.MODEL.N_FRAMES
        self.g_conv_dim = config.MODEL.G_CONV_DIM
        self.d_conv_dim = config.MODEL.D_CONV_DIM
        self.lr_schr = config.TRAIN.LR_SCHR

        self.lambda_gp = config.TRAIN.LAMBDA_GP
        self.total_epoch = config.TRAIN.TOTAL_EPOCH
        self.d_iters = config.TRAIN.D_ITERS
        self.g_iters = config.TRAIN.G_ITERS
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_workers = config.TRAIN.NUM_WORKERS
        self.g_lr = config.TRAIN.G_LR
        self.d_lr = config.TRAIN.D_LR
        self.lr_decay = config.TRAIN.LR_DECAY
        self.beta1 = config.TRAIN.BETA1
        self.beta2 = config.TRAIN.BETA2
        self.pretrained_model = config.TRAIN.PRETRAIN

        self.n_class = config.DATASET.N_CLASS
        self.k_sample = config.DATASET.K_SAMPLE
        self.dataset = config.DATASET.NAME
        self.use_tensorboard = config.OTHER.USE_TENSORBOARD
        self.test_batch_size = config.OTHER.TEST_BATCH_SIZE

        # MOCO GAN
        self.dim_z_content = config.MODEL.DIM_Z_CONTENT
        self.dim_z_motion = config.MODEL.DIM_Z_MOTION
        self.dim_z_category = config.MODEL.DIM_Z_CATEGORY
        self.video_length = config.MODEL.VIDEO_LENGTH
        self.input_channels = config.MODEL.INPUT_CHANNELS
        self.use_noise = config.MODEL.USE_NOISE
        self.noise_sigma = config.MODEL.NOISE_SIGMA

        # path
        self.image_path = config.DATASET.IMAGE_PATH
        self.log_path = config.LOG.LOG_PATH
        self.model_save_path = config.LOG.MODEL_SAVE_PATH
        self.sample_path = config.LOG.SAMPLE_PATH

        # epoch size
        self.log_epoch = config.LOG.LOG_EPOCH
        self.sample_epoch = config.LOG.SAMPLE_EPOCH
        self.model_save_epoch = config.LOG.MODEL_SAVE_EPOCH
        self.version = config.VERSION

        # Path
        self.log_path = os.path.join(self.log_path, self.version)
        self.sample_path = os.path.join(self.sample_path, self.version)
        self.model_save_path = os.path.join(self.model_save_path, self.version)

        self.device, self.parallel, self.gpus = set_device(config)
        # build model
        self.build_model()

        # set criterion FOR MOCOGAN
        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def build_model(self):
        """ Create Generator and Discriminator
        """
        print("=" * 30, '\nBuild_model...')
        self.G = VideoGenerator(self.input_channels, self.dim_z_content, self.dim_z_category, 
                                    self.dim_z_motion, self.video_length)
        self.D_s = PatchImageDiscriminator(self.input_channels, use_noise=self.use_noise, noise_sigma=self.noise_sigma)
        self.D_t = PatchVideoDiscriminator(self.input_channels, use_noise=self.use_noise, noise_sigma=self.noise_sigma)
        # if use GPU
        if self.device != "cpu":
            self.G.cuda()
            self.D_s.cuda()
            self.D_t.cuda()
        # when have multiple GPU
        if self.parallel:
            print('Use parallel...')
            print('gpus:', os.environ["CUDA_VISIBLE_DEVICES"])
            self.G = nn.DataParallel(self.G, device_ids=self.gpus)
            self.D_s = nn.DataParallel(self.D_s, device_ids=self.gpus)
            self.D_t = nn.DataParallel(self.D_t, device_ids=self.gpus)
        # Create training schedule
        self.select_opt_schr()

    def select_opt_schr(self):
        """ Create training schedule
        """
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            (self.beta1, self.beta2))
        self.ds_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_s.parameters()), self.d_lr,
                                             (self.beta1, self.beta2))
        self.dt_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_t.parameters()), self.d_lr,
                                             (self.beta1, self.beta2))
        if self.lr_schr == 'const':
            self.g_lr_scher = StepLR(self.g_optimizer, step_size=10000, gamma=1)
            self.ds_lr_scher = StepLR(self.ds_optimizer, step_size=10000, gamma=1)
            self.dt_lr_scher = StepLR(self.dt_optimizer, step_size=10000, gamma=1)
        elif self.lr_schr == 'step':
            self.g_lr_scher = StepLR(self.g_optimizer, step_size=500, gamma=0.98)
            self.ds_lr_scher = StepLR(self.ds_optimizer, step_size=500, gamma=0.98)
            self.dt_lr_scher = StepLR(self.dt_optimizer, step_size=500, gamma=0.98)
        elif self.lr_schr == 'exp':
            self.g_lr_scher = ExponentialLR(self.g_optimizer, gamma=0.9999)
            self.ds_lr_scher = ExponentialLR(self.ds_optimizer, gamma=0.9999)
            self.dt_lr_scher = ExponentialLR(self.dt_optimizer, gamma=0.9999)
        elif self.lr_schr == 'multi':
            self.g_lr_scher = MultiStepLR(self.g_optimizer, [10000, 30000], gamma=0.3)
            self.ds_lr_scher = MultiStepLR(self.ds_optimizer, [10000, 30000], gamma=0.3)
            self.dt_lr_scher = MultiStepLR(self.dt_optimizer, [10000, 30000], gamma=0.3)
        else:
            self.g_lr_scher = ReduceLROnPlateau(self.g_optimizer, mode='min',
                                                factor=self.lr_decay, patience=100,
                                                threshold=0.0001, threshold_mode='rel',
                                                cooldown=0, min_lr=1e-10, eps=1e-08,
                                                verbose=True
                            )
            self.ds_lr_scher = ReduceLROnPlateau(self.ds_optimizer, mode='min',
                                                 factor=self.lr_decay, patience=100,
                                                 threshold=0.0001, threshold_mode='rel',
                                                 cooldown=0, min_lr=1e-10, eps=1e-08,
                                                 verbose=True
                             )
            self.dt_lr_scher = ReduceLROnPlateau(self.dt_optimizer, mode='min',
                                                 factor=self.lr_decay, patience=100,
                                                 threshold=0.0001, threshold_mode='rel',
                                                 cooldown=0, min_lr=1e-10, eps=1e-08,
                                                 verbose=True
                             )


    def D_train(self, discriminator, real_batch, fake_batch, opt, scheduler, category = None):
        opt.zero_grad()
        real_labels, real_category = discriminator(real_batch)
        fake_labels, fake_category = discriminator(fake_batch)

        ones = torch.ones_like(real_labels)
        zeros = torch.ones_like(fake_labels)

        d_loss = self.gan_criterion(real_labels, ones) + self.gan_criterion(fake_labels, zeros)

        if category is not None:
            # ask the video discriminator to learn categories from training videos
            category = Variable()
            d_loss += self.category_criterion(real_category.squeeze(), category)
        
        d_loss.backward()
        opt.step()
        scheduler.step()
        
        return d_loss

    
    def G_train(self, image_discriminator, video_discriminator,
                        fake_images, fake_videos, generated_video_categories ,opt,scheduler, category = False):
        opt.zero_grad()
        # train on images
        fake_labels, fake_category = image_discriminator(fake_images)
        all_ones = torch.ones_like(fake_labels)

        g_loss = self.gan_criterion(fake_labels, all_ones)

        # train on videos
        fake_labels, fake_category = video_discriminator(fake_videos)
        all_ones = torch.ones_like(fake_labels)

        g_loss += self.gan_criterion(fake_labels, all_ones)

        if category:
            g_loss += self.category_criterion(fake_category.squeeze(), generated_video_categories)

        g_loss.backward()
        opt.step()
        scheduler.step()
        
        return g_loss


    def gen_real_video(self, data_iter):

        try:
            real_videos, real_labels = next(data_iter)
        except:
            data_iter = iter(self.data_loader)
            real_videos, real_labels = next(data_iter)
            self.epoch += 1

        return real_videos.to(self.device), real_labels.to(self.device)


    def epoch2step(self):

        self.epoch = 0
        step_per_epoch = len(self.data_loader)
        print("steps per epoch:", step_per_epoch)

        self.total_step = self.total_epoch * step_per_epoch
        self.log_step = self.log_epoch * step_per_epoch
        self.sample_step = self.sample_epoch * step_per_epoch
        self.model_save_step = self.model_save_epoch * step_per_epoch

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        self.epoch2step()
        print('total class:',self.n_class)
        # this part to check the learning process of a certain input
        fixed_z = torch.randn(self.test_batch_size * self.n_class, self.z_dim).to(self.device)
        fixed_label = torch.tensor([i for i in range(self.n_class) for j in range(self.test_batch_size)]).to(self.device)

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 1

        # Start time
        print("=" * 30, "\nStart training...")
        start_time = time.time()
        # image discriminator
        self.D_s.train()
        # video discriminator
        self.D_t.train()
        # generator
        self.G.train()
        step = start

        # for step in range(start, self.total_step + 1):
        while (step < self.total+step + 1):
            # ================ update D d_iters times ================ #
            for i in range(self.d_iters):
                # get real label
                try:
                    real_videos, real_labels = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    real_videos, real_labels = next(data_iter)
                    self.epoch += 1

                fake

                real_videos = real_videos.to(self.device)
                real_labels = real_labels.to(self.device)
                # train image discriminator
                ds_loss = self.D_train(self.D_s, real_videos, )
                step += 1
                

            # ==================== update G g_iters time ==================== #
            fake_videos_sample = sample_k_frames(fake_videos, self.n_frames, self.k_sample)
            fake_videos_downsample = vid_downsample(fake_videos)
            g_loss, g_s_loss, g_t_loss = self.G_train(fake_videos_sample, fake_videos_downsample, z_class)

            # ==================== print & save part ==================== #
            # Print out log info
            if step % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                start_time = time.time()
                # log_str = "Epoch: [%d/%d], Step: [%d/%d], time: %s, ds_loss: %.4f, dt_loss: %.4f, g_s_loss: %.4f, g_t_loss: %.4f, g_loss: %.4f, lr: %.2e" % \
                #     (self.epoch, self.total_epoch, step, self.total_step, elapsed, ds_loss, dt_loss, g_s_loss, g_t_loss, g_loss, self.g_lr_scher.get_lr()[0])
                log_str = "Epoch: [%d/%d], Step: [%d/%d], time: %s, ds_loss: %.4f, dt_loss: %.4f, g_s_loss: %.4f, g_t_loss: %.4f, g_loss: %.4f, lr: %.2e" % \
                    (self.epoch, self.total_epoch, step, self.total_step, elapsed, ds_loss, dt_loss, g_s_loss, g_t_loss, g_loss, self.g_lr_scher.get_last_lr()[0])

                if self.use_tensorboard is True:
                    write_log(self.writer, log_str, step, ds_loss_real, ds_loss_fake, ds_loss, dt_loss_real, dt_loss_fake, dt_loss, g_loss)
                print(log_str)

            # Sample images
            if step % self.sample_step == 0:
                self.G.eval()
                fake_videos = self.G(fixed_z, fixed_label)

                for i in range(self.n_class):
                    for j in range(self.test_batch_size):
                        if self.use_tensorboard is True:
                            self.writer.add_image("Class_%d_No_%d_Step_%d" % (i, j, step), make_grid(denorm(fake_videos[i * self.test_batch_size + j].data)), step)
                        else:
                            save_image(denorm(fake_videos[i * self.test_batch_size + j].data), os.path.join(self.sample_path, "Class_%d_No_%d_Step_%d.jpeg" % (i, j, step)))
                # print('Saved sample images {}_fake.png'.format(step))
                self.G.train()

            # Save model
            if step % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, 'G.pth'))
                torch.save(self.D_s.state_dict(),
                           os.path.join(self.model_save_path, 'Ds.pth'))
                torch.save(self.D_t.state_dict(),
                           os.path.join(self.model_save_path, 'Dt.pth'))
                with open(os.path.join(self.model_save_path, 'current_step.txt'),'w') as file:
                    file.write(str(step))

    def sample_image_from_videos(self, real_videos):
        #
        pass


    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)

        self.writer = SummaryWriter(log_dir=self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'G.pth')))
        self.D_s.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'Ds.pth')))
        self.D_t.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'Dt.pth')))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.ds_optimizer.zero_grad()
        self.dt_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
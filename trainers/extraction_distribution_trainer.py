import os
import math
import importlib
from tqdm import tqdm
import random
import numpy as np

import torch
from torch import autograd
import torch.distributions as dist
import torch.nn.functional as F

from loss.perceptual  import PerceptualLoss
from loss.gan import GANLoss
from loss.attn_recon import AttnReconLoss
from util.visualization import attn2image, tensor2pilimage
from util.trainer import accumulate
from trainers.base import BaseTrainer

from collections import defaultdict

class Trainer(BaseTrainer):
    def __init__(self,
                 opt,
                 net_G, net_D,
                 net_G_ema,
                 opt_G, opt_D,
                 sch_G, sch_D,
                 train_data_loader, val_data_loader=None,
                 wandb=None):
        super(Trainer, self).__init__(opt,
                                      net_G, net_D,
                                      net_G_ema,
                                      opt_G, opt_D,
                                      sch_G, sch_D,
                                      train_data_loader, val_data_loader,
                                      wandb)
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        self.seg_to_color = {}
        self.stddev_group = 4
        self.step_size = int(opt.step_size)
        height, width = opt.data.sub_path.split('-')
        self.min_size = (8, 5)
        self.load_size = (int(height), int(width))

        if getattr(self.opt.trainer, 'face_crop_method', None):
            file, crop_func = self.opt.trainer.face_crop_method.split('::')
            file = importlib.import_module(file)
            self.crop_func = getattr(file, crop_func)



    def _init_loss(self, opt):
        r"""Define training losses.

        Args:
            opt: options defined in yaml file.
        """        
        self._assign_criteria(
            'perceptual',
            PerceptualLoss(
                network=opt.trainer.vgg_param.network,
                layers=opt.trainer.vgg_param.layers,
                num_scales=getattr(opt.trainer.vgg_param, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual)

        self._assign_criteria(
            'attn_rec',
            AttnReconLoss(opt.trainer.attn_weights).to('cuda'),
            opt.trainer.loss_weight.weight_attn_rec)

        self._assign_criteria(
            'gan',
            GANLoss(opt.trainer.gan_mode).to('cuda'),
            opt.trainer.loss_weight.weight_gan)

        self._assign_criteria(
            'step',
            torch.nn.NLLLoss(),
            opt.trainer.loss_weight.weight_step)
        
        if getattr(opt.trainer.loss_weight, 'weight_face', 0) != 0:
            self._assign_criteria(
                'face', 
                PerceptualLoss(
                    network=opt.trainer.vgg_param.network,
                    layers=opt.trainer.vgg_param.layers,
                    num_scales=1,
                    ).to('cuda'),
                opt.trainer.loss_weight.weight_face)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight
    def preprocess_input(self, data):
        source_image, target_image = data['source_image'], data['target_image']
        source_skeleton, target_skeleton = data['source_skeleton'], data['target_skeleton']
        source_face, target_face = data['source_face_center'], data['target_face_center']

        source_image_batch = torch.cat((source_image, target_image), 0)
        target_image_batch = torch.cat((target_image, source_image), 0)
        source_skeleton_batch = torch.cat((source_skeleton, target_skeleton), 0)
        target_skeleton_batch = torch.cat((target_skeleton, source_skeleton), 0)
        source_face_batch = torch.cat((source_face, target_face), 0)
        target_face_batch = torch.cat((target_face, source_face), 0)

        return source_image_batch, target_image_batch,\
            source_skeleton_batch, target_skeleton_batch, \
            source_face_batch, target_face_batch
    def optimize_parameters(self, data):
        r"""Training step of generator and discriminator

        Args:
            data (dict): data used in the training step

        output_dict = {'input_image_steps': input_image_steps,
               'fake_image_steps': fake_image_steps,
               'gt_image_steps': gt_image_steps,
               'input_skeleton': input_skeleton,
               'steps': steps,
               'infos': infos}
        """          
        # training step of the generator        
        self.gen_losses = {}
        self.dis_losses = {}

        fake_imgs = []
        true_imgs = []
        steps = []
        infos = defaultdict(list)

        src_img, tgt_img, _, tgt_skeleton, _, tgt_face = self.preprocess_input(data)
        b, c, h, w = src_img.size()

        init_step = self.sample_timestep(b)

        device = src_img.device
        xt = self.sample_image(tgt_img, init_step).to(device)

        noise =  torch.normal(mean=0, std=1, size=(b, c, h, w)).to(device)
        init_image = self.sample_image(src_img, torch.zeros_like(init_step)).to(device) + noise
        index = init_step == 0
        xt[index] = init_image[index]

        z = xt.detach().cpu()

        for i in range(self.opt.window_size) :
            input_timestep = init_step + i

            xt, info = self.generate_fake_one_step(self.net_G,
                                                         src_img,
                                                         xt.detach(), tgt_skeleton, input_timestep)
            true_img = self.sample_image(tgt_img, input_timestep + 1)

            fake_imgs.append(xt)
            true_imgs.append(true_img)
            steps.append(input_timestep)
            for key, val in info.items():
                infos[key].append(val)

        fake_img_batch = torch.cat(fake_imgs, 0)
        true_img_batch = torch.cat(true_imgs, 0)
        steps = torch.cat(steps, 0)
        faces = tgt_face.repeat(self.opt.window_size, 1)
        for key, val in infos.items():
            infos[key] = [torch.cat(tensors, dim=0) for tensors in zip(*val)]

        self.calculate_G_loss(fake_img_batch, true_img_batch, src_img.repeat(self.opt.window_size, 1, 1, 1), steps.to(tgt_img.device), faces, infos)

        accumulate(self.net_G_ema, self.net_G_module, self.accum)
        # training step of the discriminator
        self.calculate_D_loss(fake_img_batch, true_img_batch, steps.to(tgt_img.device))

        fake_sample = torch.cat([z.cpu().detach(), torch.cat(fake_imgs, 3).cpu().detach()], 3)
        true_sample = torch.cat([src_img.cpu(), torch.cat(true_imgs, 3).cpu()], 3)
        self.train_sample = torch.cat([true_sample, fake_sample], 2)
    def calculate_G_loss(self, fake_img, gt_image, ref_image, ref_timestep, tgt_face, info):
        step_true = ref_timestep.long()
        if self.cal_gan_flag :
            fake_pred, step_pred = self.net_D(fake_img)
            g_loss = self.criteria['gan'](fake_pred, t_real=True, dis_update=False)
            step_loss = self.criteria['step'](step_pred, step_true)
            self.gen_losses["gan"] = g_loss
            self.gen_losses["step"] = step_loss

        else:
            self.gen_losses["gan"] = torch.tensor(0.0, device='cuda')
            self.gen_losses["step"] = torch.tensor(0.0, device='cuda')

        self.gen_losses["perceptual"] = self.criteria['perceptual'](fake_img, gt_image)
        self.gen_losses['attn_rec'] = self.criteria['attn_rec'](info, ref_image, gt_image)

        if 'face' in self.criteria:
            self.gen_losses['face'] = self.criteria['face'](
                self.crop_func(fake_img,
                               tgt_face),
                self.crop_func(gt_image,
                               tgt_face))
        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

    def calculate_D_loss(self, fake_img, gt_image, step):

        if self.cal_gan_flag:
            fake_pred, fake_step_pred = self.net_D(fake_img.detach())
            real_pred, real_step_pred = self.net_D(gt_image)
            step_true = step.long()
            fake_loss = self.criteria['gan'](fake_pred, t_real=False, dis_update=True)
            real_loss = self.criteria['gan'](real_pred, t_real=True,  dis_update=True)
            fake_step = self.criteria['step'](fake_step_pred, step_true)
            real_step = self.criteria['step'](real_step_pred, step_true)

            d_loss = fake_loss + real_loss + fake_step + real_step
            self.dis_losses["d"] = d_loss
            self.dis_losses["real_score"] = real_pred.mean()
            self.dis_losses["fake_score"] = fake_pred.mean()
            self.dis_losses['real_step'] = real_step
            self.dis_losses['fake_step'] = fake_step

            self.net_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if self.d_regularize_flag:
                gt_image.requires_grad = True
                real_img_aug = gt_image
                real_pred, step_pred = self.net_D(real_img_aug)

                r1_loss = self.d_r1_loss(real_pred, gt_image) + self.d_r1_loss(step_pred, gt_image)

                self.net_D.zero_grad()
                (self.opt.trainer.r1 / 2 * r1_loss * self.opt.trainer.d_reg_every + 0 * real_pred[0]).backward()

                self.opt_D.step()

                self.dis_losses["r1"] = r1_loss

    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True, allow_unused=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def _start_of_iteration(self, data, current_iteration):
        r"""processing before iteration

        Args:
            data (dict): data used in the training step
            current_iteration (int): current iteration 
        """             
        self.cal_gan_flag = current_iteration > self.opt.trainer.gan_start_iteration
        self.d_regularize_flag = current_iteration % self.opt.trainer.d_reg_every == 0
        return data

    def _get_visualizations(self, data, is_valid):
        r"""save visualizations when training the model

        Args:
            data (dict): data used in the training step
        """
        self.net_G_ema.eval()
        if is_valid :
            with torch.no_grad() :
                sample = self.generate_fake_full_step(self.net_G_ema, data)
        else :
            sample = self.train_sample

        return sample


    def test(self, data_loader, output_dir, current_iteration=-1):
        r"""inference function

        Args:
            data_loader: dataloader of the dataset
            output_dir (str): folder for saving the result images
            current_iteration (int): current iteration 
        """
        net_G = self.net_G_ema.eval()
        os.makedirs(output_dir, exist_ok=True)
        print('number of samples %d' % len(data_loader))
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration)

            with torch.no_grad():
                output_images = self.generate_fake_full_step(net_G, data, True)
            for output_image, file_name in zip(output_images, data['path']):
                fullname = os.path.join(output_dir, file_name)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                output_image.save(fullname)
        return None

    def generate_fake_full_step(self, net_G, data, is_test=False):
        gt_tgts = []
        fake_tgts = []

        src_img, tgt_img, _, tgt_skeleton, _, tgt_face = self.preprocess_input(data)
        b, c, h, w = src_img.size()

        init_step = torch.tensor([0 for _ in range(b)])
        noise =  torch.normal(mean=0, std=1, size=(b, c, h, w)).to(src_img.device)
        xt = self.sample_image(src_img, init_step) + noise

        gt_tgts.extend([src_img.cpu(), tgt_skeleton[:, :3].cpu()])
        fake_tgts.extend([xt.cpu(), tgt_skeleton[:, :3].cpu()])

        for step in range(self.opt.step_size) :
            timestep = torch.tensor([step for _ in range(b)])
            output_dict = net_G(src_img,
                                xt, tgt_skeleton, timestep)

            xt, info = output_dict['fake_image'], output_dict['info']

            gt_tgt = self.sample_image(tgt_img, timestep + 1)

            gt_tgts.append(gt_tgt.cpu())
            fake_tgts.append(xt.cpu())

        if is_test :
            return fake_tgts[-1]

        gt_sample = torch.cat(gt_tgts, 3)
        fake_sample = torch.cat(fake_tgts, 3)
        sample = torch.cat([gt_sample, fake_sample], 2)


        return sample


    def generate_fake_one_step(self, net_G,
                               src_image,
                               input_image, input_map, timestep, ):



        output_dict = net_G(src_image,
                            input_image, input_map, timestep)

        fake_img, info = output_dict['fake_image'], output_dict['info']

        return fake_img, info

    def sample_timestep(self, b, tgt_timestep=None):
        step = torch.randint(0, self.step_size - (self.opt.window_size - 1), (b,))
        assert step.max() + (self.opt.window_size - 1) < self.step_size, 'Over sampling!'
        if tgt_timestep != None:
            exponential_distribution = dist.Exponential(1)
            step = exponential_distribution.sample((b,)) + tgt_timestep + 1
            step[step > self.step_size] = self.step_size
        step[0] = 0
        return step.int()

    def sample_image(self, images, step, sampling_type='linear'):
        min_h, min_w = self.min_size
        max_h, max_w = self.load_size
        if sampling_type == 'exponential' :
            downscale_size = torch.stack([self.exponential_sampling(min_h, max_h, step), self.exponential_sampling(min_w, max_w, step)], dim=1).tolist()
        elif sampling_type == 'linear' :
            downscale_size = torch.stack([self.linear_sampling(min_h, max_h, step), self.linear_sampling(min_w, max_w, step)], dim=1).tolist()
        else :
            assert sampling_type in ['exponential', 'linear'], 'sampling image type error [exponential, linear]'

        result_batch = []
        for img, size in zip(images, downscale_size) :
            img_down = F.interpolate(img.unsqueeze(0), size = size, mode='bicubic', align_corners=True)
            img_up = F.interpolate(img_down, size = self.load_size, mode='bicubic', align_corners=True)
            result_batch.append(img_up)
        return torch.cat(result_batch, 0)

    def exponential_sampling(self, min_value, max_value, index):
        logspace_values = torch.logspace(np.log10(min_value),
                                         np.log10(max_value),
                                         self.step_size + 1)

        return torch.round(logspace_values[index.tolist()]).int()
    def linear_sampling(self, min_value, max_value, index):
        linspace_values = torch.linspace(torch.tensor(min_value),
                                         torch.tensor(max_value),
                                         self.step_size + 1)

        return torch.round(linspace_values[index.tolist()]).int()
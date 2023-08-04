import os
import math
import importlib
from tqdm import tqdm
import random
import numpy as np

import torch
from torch import autograd

from loss.perceptual  import PerceptualLoss
from loss.gan import GANLoss
from loss.attn_recon import AttnReconLoss
from util.visualization import attn2image, tensor2pilimage
from util.trainer import accumulate
from trainers.base import BaseTrainer
from generators.base_module import PositionalEncoding

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

        if getattr(self.opt.trainer, 'face_crop_method', None):
            file, crop_func = self.opt.trainer.face_crop_method.split('::')
            file = importlib.import_module(file)
            self.crop_func = getattr(file, crop_func)

        height, width = opt.data.sub_path.split('-')
        self.positional_encoding = PositionalEncoding(int(height) * int(width))

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

        output_dict = self.generate_fake_full_step(self.net_G, data)

        fake_img = torch.cat(output_dict['fake_image_steps'], 0)
        input_image_discretize = torch.cat(output_dict['input_image_steps'], 0)
        gt_image_discretize = torch.cat(output_dict['gt_image_steps'], 0)
        step = np.concatenate(output_dict['steps'], 0)

        self.calculate_G_loss(fake_img, gt_image_discretize, input_image_discretize, data, output_dict['info_steps'], step)

        accumulate(self.net_G_ema, self.net_G_module, self.accum)
        # training step of the discriminator
        self.calculate_D_loss(fake_img, gt_image_discretize, step)

    def calculate_G_loss(self, fake_img, gt_image, input_image, data, info, step):
        last_index = step == self.opt.step_size - 1

        if self.cal_gan_flag :
            fake_pred = self.net_D(fake_img[last_index])
            g_loss = self.criteria['gan'](fake_pred, t_real=True, dis_update=False)
            self.gen_losses["gan"] = g_loss
        else:
            self.gen_losses["gan"] = torch.tensor(0.0, device='cuda')

        self.gen_losses["perceptual"] = self.criteria['perceptual'](fake_img, gt_image)
        self.gen_losses['attn_rec'] = self.criteria['attn_rec'](info, input_image, gt_image)

        if 'target_face_center' in data and 'face' in self.criteria:
            source_face_center, target_face_center  = data['source_face_center'], data['target_face_center']
            target_face_center = torch.cat((target_face_center, source_face_center), 0)
            self.gen_losses['face'] = self.criteria['face'](
                self.crop_func(fake_img,
                               target_face_center),
                self.crop_func(gt_image,
                               target_face_center))
        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

    def calculate_D_loss(self, fake_img, gt_image, step):
        last_index = step == self.opt.step_size - 1
        fake_img = fake_img[last_index]
        gt_image = gt_image[last_index]
        if self.cal_gan_flag:
            fake_pred = self.net_D(fake_img.detach())
            real_pred = self.net_D(gt_image)
            fake_loss = self.criteria['gan'](fake_pred, t_real=False, dis_update=True)
            real_loss = self.criteria['gan'](real_pred, t_real=True,  dis_update=True)
            d_loss = fake_loss + real_loss
            self.dis_losses["d"] = d_loss
            self.dis_losses["real_score"] = real_pred.mean()
            self.dis_losses["fake_score"] = fake_pred.mean()

            self.net_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if self.d_regularize_flag:
                gt_subset = gt_image
                gt_subset.requires_grad = True
                real_img_aug = gt_subset
                real_pred = self.net_D(real_img_aug)
                r1_loss = self.d_r1_loss(real_pred, gt_subset)

                self.net_D.zero_grad()
                (self.opt.trainer.r1 / 2 * r1_loss * self.opt.trainer.d_reg_every + 0 * real_pred[0]).backward()

                self.opt_D.step()

                self.dis_losses["r1"] = r1_loss

    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
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

    def _get_visualizations(self, data):
        r"""save visualizations when training the model

        Args:
            data (dict): data used in the training step
        """
        self.net_G_ema.eval()
        output_dict = self.generate_fake_full_step(self.net_G_ema, data, True)

        z = output_dict['tgt_prev']
        fake_img = torch.cat(output_dict['fake_image_steps'], -1)
        input_skeleton = output_dict['input_skeleton']
        input_image_discretize = torch.cat(output_dict['input_image_steps'], 0)
        gt_image_discretize = torch.cat(output_dict['gt_image_steps'], -1)

        steps = np.concatenate(output_dict['steps'], 0)
        last_index = steps == self.opt.step_size - 1
        input_image_discretize = input_image_discretize[last_index]

        zero_pad = torch.zeros_like(input_image_discretize).cpu()
        true_sample = torch.cat([input_image_discretize.cpu(), input_skeleton[:,:3].cpu(), gt_image_discretize.cpu()], 3)
        fake_sample = torch.cat([zero_pad, z,fake_img.cpu()], 3)
        sample = torch.cat([true_sample, fake_sample], 2)
        sample = torch.cat(torch.chunk(sample, sample.size(0), 0)[:8], 2)
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
            input_skeleton = data['target_skeleton']
            input_image = data['source_image']
            with torch.no_grad():
                output_dict = net_G(
                    input_image, input_skeleton)    
            output_images = output_dict['fake_image']
            for output_image, file_name in zip(output_images, data['path']):
                fullname = os.path.join(output_dir, file_name)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                output_image.save(fullname)
        return None

    def generate_fake_full_step(self, net_G, data, is_inference = False):
        input_image_steps = []
        gt_image_steps = []
        fake_image_steps = []
        steps = []
        info_steps = defaultdict(list)

        source_image, target_image = data['source_image'], data['target_image']
        source_skeleton, target_skeleton = data['source_skeleton'], data['target_skeleton']
        input_image = torch.cat((source_image, target_image), 0)
        input_skeleton = torch.cat((target_skeleton, source_skeleton), 0)
        gt_image = torch.cat((target_image, source_image), 0)

        b, c, h, w = input_image.size()
        tgt_prev = torch.randn_like(gt_image)
        for step in range(self.opt.step_size) :
            sampling_step = np.array([step] * b)
            input_image_discretize, gt_image_discretize = self.discretized_dataset_sampling(input_image, gt_image, sampling_step)
            if is_inference :
                with torch.no_grad() :
                    fake_img, info, step = self.generate_fake(net_G, tgt_prev, input_skeleton, input_image, input_image_discretize, sampling_step)
            else :
                fake_img, info, step = self.generate_fake(net_G, tgt_prev, input_skeleton, input_image, input_image_discretize, sampling_step)

            fake_image_steps.append(fake_img)
            steps.append(step)
            for key, val in info.items() :
                info_steps[key].append(val)

            input_image_steps.append(input_image_discretize)
            gt_image_steps.append(gt_image_discretize)

            tgt_prev = fake_img
        # bone -> step0 -> step1 -> ... -> GT

        for key, val in info_steps.items():
            info_steps[key] = [torch.cat(tensors, dim=0) for tensors in zip(*val)]

        output_dict = {'input_image_steps': input_image_steps,
                       'fake_image_steps': fake_image_steps,
                       'gt_image_steps': gt_image_steps,
                       'input_skeleton': input_skeleton,
                       'steps': steps,
                       'info_steps': info_steps,
                       'tgt_prev': tgt_prev.cpu()}

        return output_dict




    def generate_fake(self, net_G, tgt_prev, tgt_bone, input_image, input_image_discretize, step):
        b, c, h, w = input_image.size()
        device = input_image.device

        prev_pos = self.positional_encoding(step).view(b, 1, h, w).to(device)
        next_pos = self.positional_encoding(step+1).view(b, 1, h, w).to(device)

        tgt_prev = torch.cat((tgt_prev+prev_pos, tgt_bone), 1)
        input_image = input_image_discretize + next_pos #torch.cat((input_image_discretize + next_pos, input_image), 1)
        output_dict = net_G(input_image, tgt_prev)

        fake_img, info = output_dict['fake_image'], output_dict['info']

        return fake_img, info, step
    def discretized_dataset_sampling(self, input_image, gt_image, step):

        input_image_discretize = self.discretized_image_sampling(input_image, step + 1)
        gt_image_discretize = self.discretized_image_sampling(gt_image, step + 1)

        return input_image_discretize, gt_image_discretize

    def step_sampling(self, b):
        step_size = self.opt.step_size
        sample_step = random.choices(population=range(step_size), weights=[0.5] + [0.5 / (step_size - 1)] * (step_size - 1), k=b)
        num_first = (np.array(sample_step) == 0).sum()

        if (not (num_first <= self.stddev_group or num_first % self.stddev_group == 0)) or num_first == 0 :
            sample_step.sort(reverse=False)
            change_nums = self.stddev_group - num_first % self.stddev_group
            sample_step[num_first : num_first + change_nums] = [0] * change_nums
            random.shuffle(sample_step)

        return np.array(sample_step)

    def discretized_image_sampling(self, img_batch, steps):
        step_size = self.opt.step_size
        dstep = 255 // step_size
        discretize_const = 255 - dstep * steps
        discretize_const = np.where(steps == step_size, 1, discretize_const)
        discretize_const = torch.tensor(discretize_const)[:, None, None, None].to(img_batch.device)

        img_tensor_denorm = (img_batch + 1) / 2 * 255
        img_tensor_discretize = img_tensor_denorm // discretize_const * discretize_const
        img_batch = (img_tensor_discretize / 255 * 2) - 1

        noise = torch.randn_like(img_batch).to(img_batch.device)
        img_batch = torch.where(torch.tensor(steps)[:, None, None, None].to(img_batch.device) == 0, noise, img_batch)

        return img_batch
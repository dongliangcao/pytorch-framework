import os
import time
import wandb
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from diffusers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from networks import build_network
from losses import build_loss
from metrics import build_metric

from .lr_scheduler import MultiStepRestartLR
from utils import get_root_logger


class BaseModel:
    """
    Base Model to be inherited by other models.
    Usually, feed_data, optimize_parameters and update_model
    need to be override.
    """

    def __init__(self, accelerator: Accelerator, opt):
        """
        Construct BaseModel.
        Args:
             opt (dict): option dictionary contains all information related to model.
        """
        self.accelerator = accelerator
        self.opt = opt
        self.device = accelerator.device
        self.is_train = opt['is_train']

        # build networks
        self.networks = OrderedDict()
        self._setup_networks()

        # move networks to device
        self._networks_to_device()

        # print networks info
        if self.accelerator.is_main_process:
            self.print_networks()

        # setup metrics
        self.metrics = OrderedDict()
        self._setup_metrics()

        # loss metrics
        self.loss_metrics = OrderedDict()

        # training setting
        if self.is_train:
            # train mode
            self.train()
            # init optimizers, schedulers and losses
            self._init_training_setting()

        # load pretrained models
        load_path = self.opt['path'].get('resume_state')
        if load_path and os.path.isfile(load_path):
            state_dict = torch.load(load_path)
            if self.opt['path'].get('resume', True):  # resume training
                self.resume_model(state_dict, net_only=False)
            else:  # only resume model for validation
                self.resume_model(state_dict, net_only=True)

    def feed_data(self, data):
        """process data"""
        pass
    
    def compute_total_loss(self):
         # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss
        return loss

    def optimize_parameters(self):
        """forward pass"""
        # compute total loss
        loss = self.compute_total_loss()

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        self.accelerator.backward(loss)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()

    def update_model_per_iteration(self):
        """update model per iteration"""
        pass

    def update_model_per_epoch(self):
        """
        Update model per epoch.
        """
        pass

    def get_current_learning_rate(self):
        """
        Get current learning rate for each optimizer

        Returns:
            [list]: lis of learning rate
        """
        return [optimizer.param_groups[0]['lr'] for optimizer in self.optimizers.values()]

    @torch.no_grad()
    def validation(self, dataloader, update=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            update (bool): update best metric and best model. Default True
        """
        pass

    def get_loss_metrics(self):
        self.loss_metrics = self._reduce_loss_dict()

        return self.loss_metrics

    def _init_training_setting(self):
        """
        Set up losses, optimizers and schedulers
        """
        # current epoch and iteration step
        self.curr_epoch = 0
        self.curr_iter = 0
        # optimizers and lr_schedulers
        self.optimizers = OrderedDict()
        self.schedulers = OrderedDict()

        # setup optimizers and schedulers
        self._setup_optimizers()
        self._setup_schedulers()

        # setup losses
        self.losses = OrderedDict()
        self._setup_losses()

        # best networks
        self.best_networks_state_dict = self._get_networks_state_dict()
        self.best_metric = None  # best metric to measure network

    def _setup_optimizers(self):
        def get_optimizer():
            if optim_type == 'Adam':
                return optim.Adam(optim_params, **train_opt['optims'][name])
            elif optim_type == 'AdamW':
                return optim.AdamW(optim_params, **train_opt['optims'][name])
            elif optim_type == 'RMSprop':
                return optim.RMSprop(optim_params, **train_opt['optims'][name])
            elif optim_type == 'SGD':
                return optim.SGD(optim_params, **train_opt['optims'][name])
            else:
                raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')

        train_opt = deepcopy(self.opt['train'])
        for name in self.networks:
            optim_params = []
            net = self.networks[name]
            for k, v in net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger(self.accelerator)
                    logger.warning(f'Parms {k} will not be optimized.')

            # no params to optimize
            if len(optim_params) == 0:
                logger = get_root_logger(self.accelerator)
                logger.info(f'Network {name} has no param to optimize. Ignore it.')
                continue

            if name in train_opt['optims']:
                optim_type = train_opt['optims'][name].pop('type')
                optimizer = get_optimizer()
                optimizer = self.accelerator.prepare(optimizer)
                self.optimizers[name] = optimizer
            else:
                # not optimize the network
                logger = get_root_logger(self.accelerator)
                logger.warning(f'Network {name} will not be optimized.')

    def _setup_schedulers(self):
        """
        Set up lr_schedulers
        """
        train_opt = deepcopy(self.opt['train'])
        scheduler_opts = train_opt['schedulers']
        for name, optimizer in self.optimizers.items():
            scheduler_type = scheduler_opts[name].pop('type')
            if scheduler_type in 'MultiStepRestartLR':
                scheduler = MultiStepRestartLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'CosineAnnealingLR':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'CosineAnnealingWarmRestarts':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                       **scheduler_opts[name])
            elif scheduler_type == 'OneCycleLR':
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'StepLR':
                scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'MultiStepLR':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'ExponentialLR':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'CosineSchedulerWithWarmup':
                scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_opts[name])
            elif scheduler_type == 'none':
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
            self.schedulers[name] = self.accelerator.prepare(scheduler)

    def _setup_losses(self):
        """
        Setup losses
        """
        train_opt = deepcopy(self.opt['train'])
        if 'losses' in train_opt:
            for name, loss_opt in train_opt['losses'].items():
                self.losses[name] = build_loss(self.accelerator, loss_opt)
        else:
            logger = get_root_logger(self.accelerator)
            logger.info('No loss is registered!')

    def _setup_metrics(self):
        """
        Set up metrics for data validation
        """
        val_opt = deepcopy(self.opt['val'])
        if val_opt and 'metrics' in val_opt:
            for name, metric_opt in val_opt['metrics'].items():
                self.metrics[name] = build_metric(self.accelerator, metric_opt)
        else:
            logger = get_root_logger(self.accelerator)
            logger.info('No metric is registered!')

    def _setup_networks(self):
        """
        Set up networks
        """
        for name, network_opt in deepcopy(self.opt['networks']).items():
            self.networks[name] = build_network(self.accelerator, network_opt)

    def _networks_to_device(self):
        """
        Move networks to device.
        It warps networks with DistributedDataParallel or DataParallel.
        """
        for name, network in self.networks.items():
            network = self.accelerator.prepare(network)
            self.networks[name] = network

    def _get_bare_net(self, net):
        """
        Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        net = self.accelerator.unwrap_model(net)
        return net

    def _get_networks_state_dict(self):
        """
        Get networks state dict.
        """
        state_dict = dict()
        for name in self.networks:
            state_dict[name] = deepcopy(self._get_bare_net(self.networks[name]).state_dict())

        return state_dict

    def print_networks(self):
        """
        Print the str and parameter number of networks
        """
        for net in self.networks.values():
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
            else:
                net_cls_str = f'{net.__class__.__name__}'

            bare_net = self._get_bare_net(net)
            net_params = sum(map(lambda x: x.numel(), bare_net.parameters()))

            logger = get_root_logger(self.accelerator)
            logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')

    def train(self):
        """
        Setup networks to train mode
        """
        self.is_train = True
        for name in self.networks:
            self.networks[name].train()

    def eval(self):
        """
        Setup networks to eval mode
        """
        self.is_train = False
        for name in self.networks:
            self.networks[name].eval()

    def save_model(self, net_only=False, best=False):
        """
        Save model during training, which will be used for resuming.
        Args:
            net_only (bool): only save the network state dict. Default False.
            best (bool): save the best model state dict. Default False.
        """
        if not self.accelerator.is_main_process:
            return

        if best:
            networks_state_dict = self.best_networks_state_dict
        else:
            networks_state_dict = self._get_networks_state_dict()

        if net_only:
            state_dict = {'networks': networks_state_dict}
            save_filename = 'final.pth'
        else:
            state_dict = {
                'networks': networks_state_dict,
                'epoch': self.curr_epoch,
                'iter': self.curr_iter,
                'optimizers': {},
                'schedulers': {}
            }

            for name in self.optimizers:
                state_dict['optimizers'][name] = self.optimizers[name].state_dict()

            for name in self.schedulers:
                state_dict['schedulers'][name] = self.schedulers[name].state_dict()

            save_filename = f'{self.curr_iter}.pth'

        save_path = os.path.join(self.opt['path']['models'], save_filename)

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(state_dict, save_path)
            except Exception as e:
                logger = get_root_logger(self.accelerator)
                logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def resume_model(self, resume_state, net_only=False, verbose=True):
        """Reload the net, optimizers and schedulers.

        Args:
            resume_state (dict): Resume state.
            net_only (bool): only resume the network state dict. Default False.
            verbose (bool): print the resuming process
        """
        networks_state_dict = resume_state['networks']

        # resume networks
        for name in self.networks:
            if len(list(self.networks[name].parameters())) == 0:
                if verbose:
                    logger = get_root_logger(self.accelerator)
                    logger.info(f'Network {name} has no param. Ignore it.')
                continue
            if name not in networks_state_dict:
                if verbose:
                    logger = get_root_logger(self.accelerator)
                    logger.warning(f'Network {name} cannot be resumed.')
                continue

            net_state_dict = networks_state_dict[name]
            # remove unnecessary 'module.'
            net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

            self._get_bare_net(self.networks[name]).load_state_dict(net_state_dict)

            if verbose:
                logger = get_root_logger(self.accelerator)
                logger.info(f"Resuming network: {name}")

        # resume optimizers and schedulers
        if not net_only:
            optimizers_state_dict = resume_state['optimizers']
            schedulers_state_dict = resume_state['schedulers']
            for name in self.optimizers:
                if name not in optimizers_state_dict:
                    if verbose:
                        logger = get_root_logger(self.accelerator)
                        logger.warning(f'Optimizer {name} cannot be resumed.')
                    continue
                self.optimizers[name].load_state_dict(optimizers_state_dict[name])
            for name in self.schedulers:
                if name not in schedulers_state_dict:
                    if verbose:
                        logger = get_root_logger(self.accelerator)
                        logger.warning(f'Scheduler {name} cannot be resumed.')
                    continue
                self.schedulers[name].load_state_dict(schedulers_state_dict[name])

            # resume epoch and iter
            self.curr_iter = resume_state['iter']
            self.curr_epoch = resume_state['epoch']
            if verbose:
                logger = get_root_logger(self.accelerator)
                logger.info(f"Resuming training from epoch: {self.curr_epoch}, " f"iter: {self.curr_iter}.")

    @torch.no_grad()
    def _reduce_loss_dict(self):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        """
        keys = []
        losses = []
        for name, value in self.loss_metrics.items():
            keys.append(name)
            losses.append(value)
        losses = self.accelerator.gather_for_metrics(losses)
        loss_dict = {key: loss.mean() for key, loss in zip(keys, losses)}

        return loss_dict

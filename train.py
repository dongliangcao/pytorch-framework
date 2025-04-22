import datetime
import sys
import time
import wandb
import warnings
warnings.filterwarnings("ignore")
from os import path as osp

import torch.cuda
from accelerate import Accelerator

from datasets import build_dataloader, build_dataset

from models import build_model
from utils import (AvgTimer, MessageLogger, CodeSnapshotCallback, get_env_info, get_root_logger,
                   init_wandb)
from utils.options import dict2str, parse_options



def create_train_val_dataloader(accelerator: Accelerator, opt, logger):
    train_set, val_set = None, None
    # create train and val datasets
    for dataset_name, dataset_opt in opt['datasets'].items():
        if isinstance(dataset_opt, int):  # batch_size, num_worker
            continue
        if dataset_name.startswith('train'):
            if train_set is None:
                train_set = build_dataset(accelerator, dataset_opt)
            else:
                train_set += build_dataset(accelerator, dataset_opt)
        elif dataset_name.startswith('val') or dataset_name.startswith('test'):
            if val_set is None:
                val_set = build_dataset(accelerator, dataset_opt)
            else:
                val_set += build_dataset(accelerator, dataset_opt)

    # create train and val dataloaders
    assert opt['datasets']['batch_size'] % opt['num_gpu'] == 0, f'cannot evenly split {opt["datasets"]["batch_size"]} batches to {opt["num_gpu"]} gpus'
    opt['datasets']['batch_size'] = opt['datasets']['batch_size'] // opt['num_gpu']
    train_loader = build_dataloader(
        train_set,
        opt['datasets'],
        'train',
        rank=accelerator.process_index,
        sampler=None,
        seed=opt['manual_seed'])
    batch_size = opt['datasets']['batch_size']
    train_loader = accelerator.prepare(train_loader)
    num_iter_per_epoch = len(train_loader)
    total_epochs = int(opt['train']['total_epochs'])
    total_iters = total_epochs * num_iter_per_epoch
    logger.info('Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tBatch size: {batch_size}'
                f'\n\tWorld size (gpu number): {opt["num_gpu"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    val_loader = build_dataloader(
        val_set, opt['datasets'], 'val', rank=accelerator.process_index, sampler=None,
        seed=opt['manual_seed'])
    val_loader = accelerator.prepare_data_loader(val_loader)
    logger.info('Validation statistics:'
                f'\n\tNumber of val images: {len(val_set)}')

    return train_loader, val_loader, total_epochs, total_iters


def train_pipeline(root_path, accelerator: Accelerator):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, accelerator, is_train=True)
    opt['root_path'] = root_path
    opt['use_wandb'] = opt.get('wandb_project') is not None

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(accelerator=accelerator, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    if accelerator.is_main_process:
        # initialize wandb
        init_wandb(opt)

        # save code snapshot
        CodeSnapshotCallback(opt['path']['snapshot']).on_fit_start()

    # create train and validation dataloaders
    result = create_train_val_dataloader(accelerator, opt, logger)
    train_loader, val_loader, total_epochs, total_iters = result
    opt['train']['total_iter'] = total_iters
    
    # create model
    model = build_model(accelerator, opt)

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(accelerator, opt, model.curr_iter, opt['use_wandb'])

    # training
    logger.info(f'Start training from epoch: {model.curr_epoch}, iter: {model.curr_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    try:
        while model.curr_epoch < total_epochs:
            model.curr_epoch += 1
            for train_data in train_loader:
                data_timer.record()

                model.curr_iter += 1

                # process data and forward pass
                model.feed_data(train_data)
                # backward pass
                model.optimize_parameters()
                # update model per iteration
                model.update_model_per_iteration()

                iter_timer.record()
                if model.curr_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()
                # log
                if model.curr_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': model.curr_epoch, 'iter': model.curr_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_loss_metrics())
                    msg_logger(log_vars)

                # save models and training states
                if model.curr_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save_model(net_only=False, best=False)

                # validation
                if opt.get('val') is not None and (model.curr_iter % opt['val']['val_freq'] == 0):
                    logger.info('Start validation.')
                    torch.cuda.empty_cache()
                    model.validation(val_loader)

                # synchronize all processes
                accelerator.wait_for_everyone()

                data_timer.start()
                iter_timer.start()
                # end of iter
            # update model per epoch
            model.update_model_per_epoch()
            # end of epoch
        # end of training
    except KeyboardInterrupt:
        # save the current model
        logger.info('Keyboard interrupt. Save model and exit...')
        model.save_model(net_only=False, best=False)
        model.save_model(net_only=True, best=True)
        if opt['use_wandb']:
            wandb.finish()
        sys.exit(0)

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info(f'Last Validation.')
    if opt.get('val') is not None:
        model.validation(val_loader)
    logger.info('Save the best model.')
    model.save_model(net_only=True, best=True)  # save the best model

    if opt['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))

    # init accelerator
    accelerator = Accelerator()

    # start training pipeline
    train_pipeline(root_path, accelerator)
# logging utils for training and validation
import datetime
import logging
import time
import wandb

from accelerate import Accelerator

# initialized logger
initialized_logger = {}

def init_wandb(opt):
    run = wandb.init(
        # Set the project where this run will be logged
        project=opt['wandb_project'],
        # Track hyperparameters and run metadata
        config=opt,
    )
    wandb.run.name = opt["name"]

class AvgTimer:
    """
    Timer to record the average elapsed time.

    Usage:
        timer = AvgTimer()
        for _ in range(100):
            timer.start()
            ... # do something
            timer.record()
            print(timer.get_current_time()) # print current elapsed time
        print(timer.get_avg_time()) # print average elapsed time
    """

    def __init__(self, window=200):
        """
        Args:
            window (int, optional): Sliding window to compute average time. Default 200.
        """
        self.window = window
        self.current_time = 0.
        self.total_time = 0.
        self.avg_time = 0.
        self.count = 0
        self.start()

    def start(self):
        self.start_time = time.time()

    def record(self):
        self.count += 1
        # calculate current time
        self.current_time = time.time() - self.start_time
        # calculate total time
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset timer
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger:
    """
    Message Logger

    Args:
        opt (dict): Config dict. It contains the following keys:
            name (str): experiment name.
            logger (dict): Contains 'print_freq' as logging interval.
            train (dict): Contains 'total_iter' as total iterations.

        start_iter (int, optional): Start iteration number. Default 1.
        use_wandb (bool, optional): Use Wandb. Default False.
    """

    def __init__(self, accelerator: Accelerator, opt, start_iter=1, use_wandb=False):
        self.exp_name = opt['name']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_wandb = use_wandb
        self.start_time = time.time()
        self.logger = get_root_logger(accelerator)
        self.accelerator = accelerator

    def reset_start_time(self):
        """
        Reset start time.
        """
        self.start_time = time.time()

    def __call__(self, log_dict):
        """
        Logging message

        Args:
            log_dict (dict): logging dictionary with the following keys:
                epoch (int): Current epoch.
                iter (int): Current iteration.
                lrs (list): List of learning rates.
                time (float): Elapsed time for one iteration.
                data_time (float): Elapsed time of data fetch for one iteration.
        """
        # print only in main process
        if not self.accelerator.is_main_process:
            return

        # epoch, iter, learning rates
        epoch = log_dict.pop('epoch')
        current_iter = log_dict.pop('iter')
        lrs = log_dict.pop('lrs')

        # format message
        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')]'

        # time and estimated time
        if 'time' in log_dict.keys():
            iter_time = log_dict.pop('time')
            data_time = log_dict.pop('data_time')
            # compute the total time
            total_time = time.time() - self.start_time
            # estimate the average time for one iteration
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            # estimate the rest time for the whole training
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            # add the estimated time to message
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})]'

        # other items, for example losses
        for k, v in log_dict.items():
            message += f'{k}: {v:.4e} '
            # add to tensorboard logger
            if self.use_wandb:
                # loss starts with "l_"
                if k.startswith('l_'):
                    wandb.log({f'losses/{k}': v}, step=current_iter)
                else:
                    wandb.log({f'{k}': v}, step=current_iter)

        # print message
        self.logger.info(message)

def get_root_logger(accelerator: Accelerator, logger_name='root_logger', log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str, optional): root logger name. Default: 'root_logger'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger. Default None.
        log_level (int, optional): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time. Default logging.INFO.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it.
    if logger_name in initialized_logger:
        return logger

    # initialize stream handler
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    # initialize logger level for each process
    if not accelerator.is_main_process:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # add logger to initialized logger
    initialized_logger[logger_name] = True

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision
    import platform

    msg = ('\nVersion Information: '
           f'\n\tPython: {platform.python_version()}'
           f'\n\tPyTorch: {torch.__version__}'
           f'\n\tTorchVision: {torchvision.__version__}')
    return msg
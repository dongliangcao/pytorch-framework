from tqdm import tqdm
import torch
import wandb
from accelerate import Accelerator

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_numpy
from utils import get_root_logger


@MODEL_REGISTRY.register()
class ClassificationModel(BaseModel):
    def __init__(self, accelerator: Accelerator, opt):
        super(ClassificationModel, self).__init__(accelerator, opt)

    def feed_data(self, data):
        image, target = data
        pred = self.networks['classifier'](image)

        self.loss_metrics['l_cls'] = self.losses['ce_loss'](pred, target)

    def validate_single(self, data):
        image, target = data
        pred = self.networks['classifier'](image)

        self.loss_metrics['acc'] = self.metrics['acc'](pred, target)
        self.get_loss_metrics()

    @torch.no_grad()
    def validation(self, dataloader, update=True):
        self.eval()
        self.loss_metrics.clear()
        # save results
        acc = []
        # one iteration
        if self.accelerator.is_main_process:
            pbar = tqdm(dataloader)
        else:
            pbar = dataloader

        for data in pbar:
            self.validate_single(data)
            if self.accelerator.is_main_process:
                # display loss
                pbar.set_postfix({k: v.cpu().item() for k, v in self.loss_metrics.items()})

                acc += [self.loss_metrics['acc']]

        if self.accelerator.is_main_process:
            avg_acc = to_numpy(torch.stack(acc).mean())

            if self.opt['use_wandb']:
                wandb.log({'val accuracy': avg_acc*100}, step=self.curr_iter)

            logger = get_root_logger(self.accelerator)
            logger.info(f'Val accuracy: {avg_acc*100:.2f}')

            # update best model state dict
            if update and (self.best_metric is None or (avg_acc > self.best_metric)):
                self.best_metric = avg_acc
                self.best_networks_state_dict = self._get_networks_state_dict()
                logger.info(f'Best model is updated, average geodesic error: {self.best_metric:.4f}')
        self.loss_metrics.clear()
        self.train()
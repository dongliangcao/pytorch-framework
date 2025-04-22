import warnings
warnings.filterwarnings("ignore")
from os import path as osp
from accelerate import Accelerator

from datasets import build_dataloader, build_dataset
from models import build_model
from utils import CodeSnapshotCallback, get_env_info, get_root_logger, get_time_str
from utils.options import dict2str, parse_options


def test_pipeline(root_path, accelerator: Accelerator):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, accelerator, is_train=False)
    opt['use_wandb'] = opt.get('wandb_project') is not None

    # initialize loggers
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(accelerator=accelerator, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    if accelerator.is_main_process:
        # save code snapshot
        CodeSnapshotCallback(opt['path']['snapshot']).on_fit_start()

    # create test dataset and dataloader
    test_set = None
    for _, dataset_opt in sorted(opt['datasets'].items()):
        if isinstance(dataset_opt, int):  # batch_size, num_worker
            continue
        if test_set is None:
            test_set = build_dataset(accelerator, dataset_opt)
        else:
            test_set += build_dataset(accelerator, dataset_opt)
    test_loader = build_dataloader(
        test_set, opt['datasets'], phase='val', 
        rank=accelerator.process_index, 
        sampler=None, 
        seed=opt['manual_seed'])
    test_loader = accelerator.prepare(test_loader)
    logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")

    # create model
    model = build_model(accelerator, opt)

    test_set_name = test_loader.dataset.__class__.__name__
    logger.info(f'Testing {test_set_name}...')
    model.validation(test_loader, update=False)

    # synchronize
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))

    # init accelerator
    accelerator = Accelerator()

    # start test pipeline
    test_pipeline(root_path, accelerator)

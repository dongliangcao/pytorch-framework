import importlib
import os.path as osp

from accelerate import Accelerator
from utils import get_root_logger, scandir
from utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]


def build_model(accelerator: Accelerator, opt):
    """
    Build model from options
    Args:
        opt (dict): Configuration dict. It must contain:
            type (str): Model type.

    returns:
        model (BaseModel): model built by opt.
    """
    model = MODEL_REGISTRY.get(opt['type'])(accelerator, opt)
    logger = get_root_logger(accelerator=accelerator)
    logger.info(f'Model [{model.__class__.__name__}] is created.')

    return model

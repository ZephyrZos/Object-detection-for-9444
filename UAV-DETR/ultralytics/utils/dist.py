# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    # Enhanced type checking for trainer object
    if isinstance(trainer, str):
        raise TypeError(f"Expected trainer object, got string: {trainer}")
    if not hasattr(trainer, '__class__'):
        raise TypeError(f"Expected trainer object with __class__ attribute, got: {type(trainer)}")
    
    # Additional safety check for __module__ attribute
    if not hasattr(trainer.__class__, '__module__'):
        raise TypeError(f"Trainer class {trainer.__class__} has no __module__ attribute")
    
    try:
        module, name = f'{trainer.__class__.__module__}.{trainer.__class__.__name__}'.rsplit('.', 1)
    except AttributeError as e:
        raise TypeError(f"Failed to access __module__ attribute: {e}. Trainer object may be corrupted.")
    except Exception as e:
        raise TypeError(f"Unexpected error during trainer serialization: {e}")

    content = f'''overrides = {vars(trainer.args)} \nif __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.train()'''
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='_temp_',
                                     suffix=f'{id(trainer)}.py',
                                     mode='w+',
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    
    # Enhanced type checking for trainer object
    if isinstance(trainer, str):
        raise TypeError(f"Expected trainer object, got string: {trainer}")
    if not hasattr(trainer, '__class__'):
        raise TypeError(f"Expected trainer object with __class__ attribute, got: {type(trainer)}")
    
    # Additional safety check for __module__ attribute
    if not hasattr(trainer.__class__, '__module__'):
        raise TypeError(f"Trainer class {trainer.__class__} has no __module__ attribute")
    
    try:
        # Test access to __module__ attribute
        _ = trainer.__class__.__module__
    except AttributeError as e:
        raise TypeError(f"Failed to access __module__ attribute: {e}. Trainer object may be corrupted.")
    except Exception as e:
        raise TypeError(f"Unexpected error during trainer serialization: {e}")
    
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = str(Path(sys.argv[0]).resolve())
    safe_pattern = re.compile(r'^[a-zA-Z0-9_. /\\-]{1,128}$')  # allowed characters and maximum of 100 characters
    if not (safe_pattern.match(file) and Path(file).exists() and file.endswith('.py')):  # using CLI
        file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{world_size}', '--master_port', f'{port}', file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f'{id(trainer)}.py' in file:  # if temp_file suffix in file
        os.remove(file)

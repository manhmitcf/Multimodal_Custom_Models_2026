import os
import random
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> int:
    """
    Set random seeds for full deterministic reproducibility across:
    Python random module, OS hash seed, NumPy, and PyTorch (CPU & CUDA).
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic master seed locked: {seed} (PyTorch, CUDA, NumPy, Random)")
    return seed


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function ensuring reproducible randomness across
    multiprocessing workers for Python random, NumPy, and PyTorch.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


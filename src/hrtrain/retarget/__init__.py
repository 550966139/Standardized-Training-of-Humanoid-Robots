"""Retargeting canonical motion → robot joint trajectory."""
from .gmr_wrapper import retarget_to_g1
from .to_train_npz import write_training_npz

__all__ = ["retarget_to_g1", "write_training_npz"]

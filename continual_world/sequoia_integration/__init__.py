""" Wrapper for the Continual-World repo that exposes their algorithms as Methods for Sequoia. """

from .base_sac_method import SAC
from .agem import AGEM
from .packnet import PackNet
from .reg_methods import L2Regularization, EWC, MAS
from .vcl import VCL

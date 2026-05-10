from .base_env import MultiAgentEnv
from .multi_uav.multi_uav_env import MultiUAVEnv

try:
    from .smacv1.smac_env import SMACWrapperEnv
except ImportError:
    SMACWrapperEnv = None

try:
    from .vec_env import BaseVecEnv, DummyVecEnv, SubprocVecEnv
except ImportError:
    BaseVecEnv = DummyVecEnv = SubprocVecEnv = None

__all__ = [
    'MultiUAVEnv',
    'SMACWrapperEnv',
    'BaseVecEnv',
    'DummyVecEnv',
    'SubprocVecEnv',
    'MultiAgentEnv',
]

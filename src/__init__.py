# -*- coding: utf-8 -*-
"""
Spot Instance Bidding Strategy using Reinforcement Learning
Tối ưu chiến lược sử dụng Spot Instance trên Cloud bằng Học tăng cường
"""

from .environment import SpotInstanceEnv
from .baselines import AlwaysOnDemand, AlwaysSpot, ThresholdPolicy, RandomPolicy
from .data_generator import SpotPriceGenerator, WorkloadGenerator

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "SpotInstanceEnv",
    "AlwaysOnDemand",
    "AlwaysSpot",
    "ThresholdPolicy",
    "RandomPolicy",
    "SpotPriceGenerator",
    "WorkloadGenerator",
]

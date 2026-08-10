#!/usr/bin/env python3
"""
running_mean_std.py — Custom RunningMeanStd Observation Normalizer

Implementation of Welford's algorithm for parallel batch mean and variance updates.
"""

from typing import Tuple, Optional
import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and running variance over streaming observations.
    Uses Welford's algorithm for parallel batch updates.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        """
        Updates mean, variance, and count from a batch of observations x.
        x can be shape (batch_size, *shape) or (*shape,).
        """
        batch_x = np.asarray(x, dtype=np.float64)
        if batch_x.ndim == len(self.shape):
            batch_x = np.expand_dims(batch_x, axis=0)

        batch_mean = np.mean(batch_x, axis=0)
        batch_var = np.var(batch_x, axis=0)
        batch_count = float(batch_x.shape[0])

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        """
        Parallel Welford update combining existing running moments with batch moments.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, obs: np.ndarray, clip_obs: float = 10.0) -> np.ndarray:
        """
        Normalizes observation array using current running mean and standard deviation.
        clip_obs bounds normalized values to [-clip_obs, clip_obs].
        """
        norm_obs = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        if clip_obs is not None:
            norm_obs = np.clip(norm_obs, -clip_obs, clip_obs)
        return norm_obs.astype(np.float32)

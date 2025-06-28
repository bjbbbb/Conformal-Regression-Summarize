# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import numpy as np
import torch

from torchcp.regression.utils.metrics import Metrics
from torchcp.utils.common import calculate_conformal_value


def correct_interval_sizes(sizes):
    N, n = sizes.shape
    m = np.argmin(sizes, axis=1)

    indices = np.arange(n)
    mask = indices[:, np.newaxis] <= m[np.newaxis, :]

    left_max = np.maximum.accumulate(sizes, axis=1)
    right_max = np.flip(np.maximum.accumulate(np.flip(sizes, axis=1), axis=1), axis=1)

    corrected_sizes = np.where(mask.T, left_max, right_max)

    return corrected_sizes

class CTI():

    def __init__(self, model, quantiles=None, upper=None, lower=None):
        # super().__init__(model)
        self._model = model
        self._metric = Metrics()
        if quantiles is None:
            quantiles = np.concatenate(([0.001], np.arange(0.025, 1, 0.025), [0.999]))
        if isinstance(quantiles, np.ndarray):
            quantiles = quantiles.tolist()
        if upper is None:
            upper = np.inf
        if lower is None:
            lower = -np.inf
        self.upper = upper
        self.lower = lower
        self.quantiles = quantiles

    def find_containing_intervals(self, tmp_predicts, tmp_labels):
        n, m = tmp_predicts.shape
        # Reshape tmp_labels to match the shape of tmp_predicts
        tmp_labels_expanded = np.tile(tmp_labels, (m - 1, 1)).T
        # Check if each label is within the corresponding intervals
        mask = np.logical_and(tmp_labels_expanded >= tmp_predicts[:, :-1],
                              tmp_labels_expanded < tmp_predicts[:, 1:])
        # Find the indices of the first True value in each row (first containing interval)
        indices = mask.argmax(axis=1)
        # Set indices to None where no containing interval is found
        indices = np.where(mask.any(axis=1), indices, -1)
        # Compute the lengths of the containing intervals
        lengths = np.where(indices != -1,
                           tmp_predicts[np.arange(n), indices + 1] - tmp_predicts[np.arange(n), indices], np.inf)

        return indices, lengths

    def calibrate(self, cal_dataloader, alpha):
        intervals_list = []
        y_truth_list = []
        for examples in cal_dataloader:
            tmp_x, tmp_labels = examples[0].numpy(), examples[1].numpy()
            tmp_predicts = self._model.predict(tmp_x, quantiles=self.quantiles)
            tmp_predicts = np.hstack([np.ones((tmp_predicts.shape[0], 1)) * self.lower, tmp_predicts, np.ones((tmp_predicts.shape[0], 1)) * self.upper])
            intervals_list.append(self.find_containing_intervals(tmp_predicts, tmp_labels)[1])

        intervals = np.concatenate(intervals_list)
        self.q_hat = calculate_conformal_value(torch.FloatTensor(intervals), alpha).numpy()

    def evaluate(self, data_loader):
        y_list = []
        predict_list = []
        for examples in data_loader:
            tmp_x, tmp_y = examples[0].numpy(), examples[1].numpy()
            tmp_predicts = self._model.predict(tmp_x, quantiles=self.quantiles)
            tmp_predicts = np.hstack([np.ones((tmp_predicts.shape[0], 1)) * self.lower, tmp_predicts,
                                      np.ones((tmp_predicts.shape[0], 1)) * self.upper])
            # Compute interval lengths
            interval_lengths = np.diff(tmp_predicts, axis=1)
            # Create a mask for intervals with length less than or equal to self.q_hat
            mask = interval_lengths <= self.q_hat
            # Create arrays for lower and upper bounds of prediction intervals
            lower_bounds = np.where(mask, tmp_predicts[:, :-1], np.nan)
            upper_bounds = np.where(mask, tmp_predicts[:, 1:], np.nan)
            # Stack lower and upper bounds along a new axis
            tmp_prediction_intervals = np.stack((lower_bounds, upper_bounds), axis=2)
            y_list.append(tmp_y)
            predict_list.append(tmp_prediction_intervals)

        y_array = np.concatenate(y_list)
        predict_array = np.concatenate(predict_list)

        # Compute coverage rate
        valid_intervals = ~np.isnan(predict_array).any(axis=2)
        covered = np.logical_and(y_array.reshape(-1, 1) >= predict_array[:, :, 0],
                                 y_array.reshape(-1, 1) <= predict_array[:, :, 1])
        coverage_rate = np.mean(np.any(covered, axis=1))

        # Compute average size of prediction intervals
        interval_sizes = np.nansum(np.diff(predict_array, axis=2), axis=1)
        average_size = np.mean(interval_sizes[np.any(valid_intervals, axis=1)])
        print(coverage_rate, average_size)

        # Create a dictionary to store the evaluation results
        res_dict = {
            "Coverage_rate": coverage_rate,
            "Average_size": average_size
        }

        return res_dict

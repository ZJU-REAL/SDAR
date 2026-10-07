# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for sample-count-weighted validation success-rate aggregation.

Covers ``_weighted_mean_success_rate`` in ``verl.trainer.ppo.ray_trainer``
(ZJU-REAL/SDAR#60): per-batch success rates must be weighted by the number
of samples in each batch, otherwise tiny trailing batches dominate the
reported validation metric.
"""

import unittest

import numpy as np

from verl.trainer.ppo.ray_trainer import _weighted_mean_success_rate


class WeightedMeanSuccessRateTest(unittest.TestCase):
    def test_equal_batch_sizes_match_plain_mean(self):
        self.assertAlmostEqual(
            _weighted_mean_success_rate([0.5, 0.25, 0.75], [32, 32, 32]),
            0.5,
        )

    def test_tiny_trailing_batch_does_not_dominate(self):
        # Bamboogle case from ZJU-REAL/SDAR#60: a 124-sample batch at 1.0
        # followed by a 1-sample batch at 0.0 must report ~0.992, not 0.5.
        self.assertAlmostEqual(
            _weighted_mean_success_rate([1.0, 0.0], [124, 1]),
            124 / 125,
        )

    def test_single_batch(self):
        self.assertAlmostEqual(_weighted_mean_success_rate([0.75], [10]), 0.75)

    def test_returns_python_float(self):
        self.assertIsInstance(_weighted_mean_success_rate([0.5], [4]), float)


if __name__ == "__main__":
    unittest.main()

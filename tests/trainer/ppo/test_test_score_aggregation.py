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
"""Tests for per-trajectory validation test_score aggregation.

Covers ``_mean_reward_per_trajectory`` in ``verl.trainer.ppo.ray_trainer``
(ZJU-REAL/SDAR#60, second half): validation batches hold one row per
active (trajectory, step) pair and every step row of a trajectory carries
the same broadcast episode reward, so averaging over all rows would weight
each trajectory by its episode length. Only the first row per ``traj_uid``
must contribute, mirroring the tool-call-count dedup in ``_validate``.
"""

import unittest

import numpy as np

from verl.trainer.ppo.ray_trainer import _mean_reward_per_trajectory


class FakeTensor:
    """Minimal stand-in for a CPU torch tensor (only ``__array__`` is used)."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)

    def __array__(self, dtype=None):
        return self._values.astype(dtype) if dtype else self._values


class MeanRewardPerTrajectoryTest(unittest.TestCase):
    def test_step_expansion_does_not_change_metric(self):
        # Table from ZJU-REAL/SDAR#60: correctness [1, 0] with episode
        # lengths [1, 4] and [4, 1] must both report 0.5. Averaging over
        # step rows would give 0.2 and 0.8 instead.
        short_long = _mean_reward_per_trajectory(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            ["bamboogle"] * 5,
            ["a", "b", "b", "b", "b"],
        )
        self.assertAlmostEqual(short_long["bamboogle"], 0.5)
        long_short = _mean_reward_per_trajectory(
            [1.0, 1.0, 1.0, 1.0, 0.0],
            ["bamboogle"] * 5,
            ["a", "a", "a", "a", "b"],
        )
        self.assertAlmostEqual(long_short["bamboogle"], 0.5)

    def test_distinct_rollouts_of_same_question_are_kept(self):
        # Validation repeats (val_kwargs.n > 1) give each rollout a distinct
        # traj_uid, so each rollout counts once: rewards 1.0 and 0.0 -> 0.5.
        result = _mean_reward_per_trajectory(
            [1.0, 1.0, 0.0, 0.0, 0.0],
            ["bamboogle"] * 5,
            ["rollout-1", "rollout-1", "rollout-2", "rollout-2", "rollout-2"],
        )
        self.assertAlmostEqual(result["bamboogle"], 0.5)

    def test_groups_by_data_source(self):
        result = _mean_reward_per_trajectory(
            [1.0, 0.0, 0.0, 1.0, 1.0],
            ["a", "a", "b", "b", "b"],
            ["u1", "u2", "u3", "u3", "u4"],
        )
        self.assertAlmostEqual(result["a"], 0.5)
        self.assertAlmostEqual(result["b"], 0.5)

    def test_accepts_numpy_and_tensor_like_inputs(self):
        result = _mean_reward_per_trajectory(
            FakeTensor([1.0, 0.0, 0.0]),
            np.array(["bamboogle"] * 3),
            np.array(["u1", "u2", "u2"], dtype=object),
        )
        self.assertAlmostEqual(result["bamboogle"], 0.5)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(_mean_reward_per_trajectory([], [], []), {})

    def test_returns_python_floats(self):
        result = _mean_reward_per_trajectory([1.0, 0.0], ["s", "s"], ["u1", "u2"])
        self.assertIsInstance(result["s"], float)


if __name__ == "__main__":
    unittest.main()

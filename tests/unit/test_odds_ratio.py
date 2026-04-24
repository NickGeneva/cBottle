# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Locks the contract of the guidance-hook path in ``_sample_with_latents``
and catches accidental regressions in the upstreamed log-odds-ratio
pipeline. Uses the same tiny CBottle3d + MockClassifier harness as
``test_inference.test_cbottle3d_sample``.
"""

import math

import pytest
import torch

from cbottle import models
from cbottle.datasets.base import BatchInfo
from cbottle.datasets.dataset_2d import MAX_CLASSES as LABEL_DIM
from cbottle.inference import CBottle3d
from cbottle.models.networks import Output


class MockClassifier:
    """Mirror of test_inference.MockClassifier with gradient-flowing logits."""

    def __init__(self):
        self.is_called = False

    def __call__(self, x_hat, *args, **kwargs):
        self.is_called = True
        logits = (
            torch.ones(1, 1, 1, 12 * 8**2, requires_grad=True).cuda() * x_hat.mean()
        )
        return Output(out=None, logits=logits)


def _make_tiny_cbottle3d(separate_classifier):
    net = models.get_model(
        models.ModelConfigV1(model_channels=8, out_channels=3, label_dim=LABEL_DIM)
    )
    net.batch_info = BatchInfo(
        channels=["rlut", "rsut", "rsds"],
        scales=[1.0, 1.0, 1.0],
        center=[0.0, 0.0, 0.0],
    )
    net.cuda()
    # channels_last=False keeps the NCHW path so autograd through the
    # healpix pad survives (same path the likelihood regression uses).
    return CBottle3d(
        net,
        sigma_min=0.02,
        sigma_max=200.0,
        num_steps=2,
        channels_last=False,
        separate_classifier=separate_classifier,
    )


def _make_batch():
    shape = (1, 3, 1, 12 * 64 * 64)
    return {
        "target": torch.randn(*shape).cuda(),
        "labels": torch.nn.functional.one_hot(
            torch.tensor([1]), num_classes=LABEL_DIM
        ).cuda(),
        "condition": torch.zeros(1, 0, 1, shape[-1]).cuda(),
        "second_of_day": torch.tensor([[43200]]).cuda(),
        "day_of_year": torch.tensor([[180]]).cuda(),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_odds_ratio_forward_only():
    classifier = MockClassifier()
    model = _make_tiny_cbottle3d(classifier)
    batch = _make_batch()
    guidance_pixels = torch.tensor([0]).cuda()

    results = model.calculate_odds_ratio(
        batch,
        guidance_pixels,
        num_steps=2,
        extra_steps_intervals=(),  # no densification at this tiny size
        divergence_samples=1,
        guidance_on=0.0,
        guidance_off=float("inf"),
        run_backward=False,
        bf16=False,
    )

    # Forward-only contract: scalar + tensor keys, no backward fields.
    assert set(results.keys()) == {
        "forward_guidance_div_integral",
        "forward_score_div_integral",
        "initial_log_prob",
        "forward_latents",
    }
    for k in (
        "forward_guidance_div_integral",
        "forward_score_div_integral",
        "initial_log_prob",
    ):
        assert isinstance(results[k], float)
        assert math.isfinite(results[k])
    assert results["forward_latents"].shape == batch["target"].shape
    assert classifier.is_called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_odds_ratio_full_three_phases():
    classifier = MockClassifier()
    model = _make_tiny_cbottle3d(classifier)
    batch = _make_batch()
    guidance_pixels = torch.tensor([0]).cuda()

    results = model.calculate_odds_ratio(
        batch,
        guidance_pixels,
        num_steps=2,
        extra_steps_intervals=(),
        divergence_samples=1,
        guidance_on=0.0,
        guidance_off=float("inf"),
        run_backward=True,
        bf16=False,
    )

    expected_scalar_keys = {
        "forward_guidance_div_integral",
        "forward_score_div_integral",
        "backward_guidance_div_integral",
        "backward_score_div_integral",
        "backward_no_guidance_guidance_div_integral",
        "backward_no_guidance_score_div_integral",
        "backward_gaussian_logp",
        "backward_no_guidance_gaussian_logp",
        "initial_log_prob",
    }
    expected_tensor_keys = {
        "forward_latents",
        "backward_latents",
        "backward_no_guidance_latents",
    }
    assert set(results.keys()) == expected_scalar_keys | expected_tensor_keys

    for k in expected_scalar_keys:
        assert isinstance(results[k], float), k
        assert math.isfinite(results[k]), k
    for k in expected_tensor_keys:
        assert isinstance(results[k], torch.Tensor), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sample_with_latents_hooks_guidance_fn_without_data():
    """guidance_fn is honored even when guidance_data is None (backward-no-guidance path).

    Locks the _call_guidance gating added alongside calculate_odds_ratio:
    passing a guidance_fn must enable the hook even with guidance_scale=0,
    so the tracker can still observe each step.
    """
    classifier = MockClassifier()
    model = _make_tiny_cbottle3d(classifier)
    batch = _make_batch()

    calls: list[float] = []

    def tracker_fn(guidance_data, logits, x_hat, denoised, t_hat):
        calls.append(float(t_hat))
        return torch.zeros_like(x_hat)

    _processed, _coords, _raw = model._sample_with_latents(
        batch,
        guidance_pixels=None,
        guidance_scale=0.0,
        bf16=False,
        return_untransformed=True,
        guidance_fn=tracker_fn,
    )

    assert calls, "tracker_fn should have been invoked at each step"

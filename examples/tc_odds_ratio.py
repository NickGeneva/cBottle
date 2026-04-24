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
"""Minimal example: classifier-guided log-odds-ratio for a TC sample."""

import cbottle.inference
import matplotlib.pyplot as plt
import pandas as pd
import torch
import warnings
from cbottle.datasets.dataset_3d import get_dataset
from cbottle.visualizations import visualize

warnings.filterwarnings("ignore", message="Cannot do a zero-copy NCHW to NHWC")

times = pd.date_range(start="2018-09-01T16:00:00", end="2018-09-01T16:00:00", freq="1h")
lons = [-80]
lats = [25]

ds = get_dataset(dataset="amip")
ds.set_times(times)
loader = torch.utils.data.DataLoader(ds, batch_size=1)
batch = next(iter(loader))

model = cbottle.inference.load("cbottle-3d-moe-tc")
guidance_pixels = model.get_guidance_pixels(lons, lats)

# Reduced-cost schedule so the example runs in a few minutes on one GPU.
# For paper-accurate numbers drop the overrides; see CBottle3d.calculate_odds_ratio.
results = model.calculate_odds_ratio(
    batch,
    guidance_pixels,
    num_steps=18,
    extra_steps_intervals=((15, 20, 5),),
    divergence_samples=1,
    compute_forward_divergences=False,
    seed=0,
)

log_odds_ratio = (
    (results["backward_no_guidance_gaussian_logp"] - results["backward_gaussian_logp"])
    + (
        results["backward_no_guidance_score_div_integral"]
        - results["backward_score_div_integral"]
    )
    - results["backward_guidance_div_integral"]
)
print(f"log p_unguided / p_guided  ~  {log_odds_ratio:.2f}")

sample = model._post_process(results["forward_latents"])
visualize(sample[0, 0, 0].detach().cpu(), nest=True, region="carib")
plt.savefig("tc_odds_ratio_sample.png", bbox_inches="tight", dpi=120)

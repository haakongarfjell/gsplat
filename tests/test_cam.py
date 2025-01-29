"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing_extensions import Literal, assert_never
import math

import pytest
import torch
import os

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_world_to_cam(test_data):
    from gsplat.cuda._torch_impl import _world_to_cam
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    viewmats = test_data["viewmats"]
    means = test_data["means"]
    scales = test_data["scales"]
    quats = test_data["quats"]
    covars, _ = quat_scale_to_covar_preci(quats, scales)
    means.requires_grad = True
    covars.requires_grad = True
    viewmats.requires_grad = True

    # forward
    means_c, covars_c = world_to_cam(means, covars, viewmats)
    _means_c, _covars_c = _world_to_cam(means, covars, viewmats)
    torch.testing.assert_close(means_c, _means_c)
    torch.testing.assert_close(covars_c, _covars_c)

    # backward
    v_means_c = torch.randn_like(means_c)
    v_covars_c = torch.randn_like(covars_c)
    v_means, v_covars, v_viewmats = torch.autograd.grad(
        (means_c * v_means_c).sum() + (covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
    )
    _v_means, _v_covars, _v_viewmats = torch.autograd.grad(
        (_means_c * v_means_c).sum() + (_covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
    )

    print(v_means)
    torch.testing.assert_close(v_means, _v_means)
    torch.testing.assert_close(v_covars, _v_covars)
    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)
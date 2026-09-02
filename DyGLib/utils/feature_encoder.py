#!/usr/bin/env python3
"""Reconstruct the on-the-fly feature transform used to build *_stats_mlp_ks_* datasets.

Mirrors SAD/feature_encoder.py so the saved encoder checkpoints
(raw_data/<data_set>.pt, MLP raw -> 128 relu -> latent) can be used in DyGLib to
attribute SHAP values to the RAW features.
"""
import numpy as np
import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """raw features -> reduced (mlp_ks) features, differentiable w.r.t. raw."""

    def __init__(self, encoder_state, fit_feats):
        super().__init__()
        self.register_buffer("w0", torch.tensor(np.asarray(encoder_state["0.weight"], dtype=np.float32)))
        self.register_buffer("b0", torch.tensor(np.asarray(encoder_state["0.bias"], dtype=np.float32)))
        self.has_hidden = "3.weight" in encoder_state
        if self.has_hidden:
            self.register_buffer("w3", torch.tensor(np.asarray(encoder_state["3.weight"], dtype=np.float32)))
            self.register_buffer("b3", torch.tensor(np.asarray(encoder_state["3.bias"], dtype=np.float32)))
            self.latent_dim = int(self.w3.shape[0])
        else:
            self.latent_dim = int(self.w0.shape[0])
        self.raw_dim = int(self.w0.shape[1])

        fit_feats = np.asarray(fit_feats, dtype=np.float64)
        if fit_feats.ndim == 1:
            fit_feats = fit_feats[None, :]
        if fit_feats.shape[1] != self.raw_dim:
            raise ValueError(
                "raw feature dim %d != encoder input dim %d" % (fit_feats.shape[1], self.raw_dim)
            )

        raw_mean = fit_feats.mean(axis=0)
        raw_std = fit_feats.std(axis=0)
        raw_scale = np.where(raw_std < 1e-6, 1.0, raw_std)

        scaled = ((fit_feats - raw_mean) / raw_scale).astype(np.float32)
        with torch.no_grad():
            z = self._mlp(torch.from_numpy(scaled)).numpy().astype(np.float64)
        lat_mu = z.mean(axis=0)
        lat_sd = z.std(axis=0)
        lat_scale = np.where(lat_sd < 1e-6, 1e-6, lat_sd)

        self.register_buffer("raw_mean", torch.tensor(raw_mean, dtype=torch.float32))
        self.register_buffer("raw_scale", torch.tensor(raw_scale, dtype=torch.float32))
        self.register_buffer("lat_mu", torch.tensor(lat_mu, dtype=torch.float32))
        self.register_buffer("lat_scale", torch.tensor(lat_scale, dtype=torch.float32))
        self.eval()

    @staticmethod
    def from_checkpoint(encoder_path, fit_feats):
        state = torch.load(encoder_path, map_location="cpu")
        if isinstance(state, dict) and "0.weight" in state and "0.bias" in state:
            return FeatureEncoder(state, fit_feats)
        raise ValueError("encoder %s does not look like a FeatureEncoder state dict" % encoder_path)

    def _mlp(self, x):
        h = torch.relu(x @ self.w0.t() + self.b0)
        return h if not self.has_hidden else h @ self.w3.t() + self.b3

    def forward(self, x):
        x = x.float()
        is_zero = x.detach().abs().sum(dim=-1, keepdim=True) == 0
        scaled = (x - self.raw_mean) / self.raw_scale
        out = (self._mlp(scaled) - self.lat_mu) / self.lat_scale
        return out.masked_fill(is_zero.expand_as(out), 0.0)
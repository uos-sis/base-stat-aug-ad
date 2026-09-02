#!/usr/bin/env python3
"""Reconstruct the on-the-fly feature transform used to build *_mlp_ks_64 JODIE
datasets.

The generator ``modus2/datasets/jodie/reduce_features_mlp.py`` turned raw JODIE
features into the reduced vectors that SAD/GraphSAGE/... consume as::

    reduced = ( MLP( StandardScaler(raw) ) - latent_mu ) / latent_sd

``MLP`` (raw_data/{data_set}_mlp_ks_64.pt, 181 -> 128 relu -> 64) is saved
as a state dict, but the input scaler and the per-dim latent mean/std are fitted over
the *whole* raw feature matrix and are not stored anywhere.  ``FeatureEncoder`` refits
those statistics here so ``encoder(raw)`` reproduces the stored reduced features, and
because it is a differentiable ``nn.Module`` it can be prepended to a model so SHAP
can attribute the raw input features.

The only special case is padding: preprocessing wrote an all-zero row for the phantom
edge/node ``0`` and SAD's ``edge_padding`` appends all-zero rows.  Those rows were
never pushed through the encoder, so ``forward`` maps all-zero input rows to all-zero
output rows (matching the stored files).
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
        """Build from the saved MLP state dict (keys '0.*', '3.*')."""
        state = torch.load(encoder_path, map_location="cpu")
        if isinstance(state, dict) and "0.weight" in state and "0.bias" in state:
            encoder_state = state
        else:
            raise ValueError(
                "encoder %s does not look like a FeatureEncoder state dict (keys '0.*' and "
                "optionally '3.*'); got: %s" % (encoder_path, list(state.keys()))
            )
        return FeatureEncoder(encoder_state, fit_feats), state

    def _mlp(self, x):
        h = torch.relu(x @ self.w0.t() + self.b0)
        if not self.has_hidden:
            return h
        return h @ self.w3.t() + self.b3

    def forward(self, x):
        """raw -> reduced (last dim maps to latent_dim); all-zero rows stay zero."""
        x = x.float()
        is_zero = x.detach().abs().sum(dim=-1, keepdim=True) == 0
        scaled = (x - self.raw_mean) / self.raw_scale
        out = (self._mlp(scaled) - self.lat_mu) / self.lat_scale
        return out.masked_fill(is_zero.expand_as(out), 0.0)

"""Configuration module for the pension-dp project.

Implements model parameters, utility definitions, and validation rules from:
Lichtenstern & Zagst (2022) - "Optimal investment strategies for pension funds
with regulation-conform dynamic pension payment management in the absence of guarantees".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np


ConfigMode = Literal["as_printed", "paper_calibrated"]


@dataclass(frozen=True)
class Config:
    """Model and numerical parameters for the pension MDP.

    Attributes:
        mode: Calibration mode.
            - "as_printed": Uses r=0.01, lambda_x=0.0118 (gives P0 approx 194 for alpha=0).
            - "paper_calibrated": Uses r=0.01, lambda_x=0.0212 such that r + lambda_x = 0.0312
              (gives P0 = 277, 270, 258 for alpha = 0.0, 0.2, 0.4 matching paper Tables 1 and 2).
        r: Risk-free rate (safe asset return), default 0.01 (paper eq. 5.1).
        mu: Expected return (drift) of risky asset, default 0.0297 (paper eq. 5.1).
        sigma: Volatility of risky asset, default 0.1175 (paper eq. 5.1).
        lambda_x: Force of mortality / cohort mortality rate.
        beta: Subjective discount rate, default 0.03 (paper eq. 5.33).
        delta: Time step in years, default 1.0 (paper eq. 5.3).
        b: HARA utility curvature parameter, default -1.0 (paper eq. 5.34).
        a_hat: HARA utility scaling parameter, default 1.0 (paper eq. 5.34).
        F: Subsistence consumption level / minimum pension floor, default 25.8 (paper eq. 5.34).
        p_bar: Regulatory corridor target coverage ratio, default 1.125 (paper eq. 5.9).
        ccr_min: Lower regulatory corridor bound, default 1.00 (paper eq. 5.9).
        ccr_max: Upper regulatory corridor bound, default 1.25 (paper eq. 5.9).
        V0: Initial wealth pot, default 10000.0 (paper Section 5).
        alpha: Buffer wealth fraction parameter in [0, 0.5], default 0.0 (paper eq. 5.4, 5.5).
        n_V: Number of wealth grid nodes, default 1000.
        V_min_factor: Lower wealth boundary multiplier (V_min = V_min_factor * V0), default 0.2.
        V_max_factor: Upper wealth boundary multiplier (V_max = V_max_factor * V0), default 5.0.
        n_CCR: Number of CCR grid nodes, default 26.
        n_Z: Number of shock grid nodes, default 40.
        n_actions: Number of discrete investment actions in [0, 1], default 21.
        P_min: Clamping lower bound on pension (must be > F), default F + 1.0 = 26.8.
        P_max: Clamping upper bound on pension, default 2500.0.
        corridor_tolerance: Numerical tolerance for CCR corridor boundaries, default 1e-9.
    """

    mode: ConfigMode = "paper_calibrated"
    r: float = 0.01
    mu: float = 0.0297
    sigma: float = 0.1175
    lambda_x: float = field(default=0.0212)
    beta: float = 0.03
    delta: float = 1.0
    b: float = -1.0
    a_hat: float = 1.0
    F: float = 25.8
    p_bar: float = 1.125
    ccr_min: float = 1.00
    ccr_max: float = 1.25
    V0: float = 10000.0
    alpha: float = 0.0
    n_V: int = 1000
    V_min_factor: float = 0.2
    V_max_factor: float = 5.0
    n_CCR: int = 26
    n_Z: int = 40
    n_actions: int = 21
    P_min: float = 26.8
    P_max: float = 2500.0
    corridor_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate configuration parameters and edge-case invariants."""
        # Adjust lambda_x based on mode if default was not explicitly custom
        if self.mode == "as_printed" and math.isclose(self.lambda_x, 0.0212):
            object.__setattr__(self, "lambda_x", 0.0118)
        elif self.mode == "paper_calibrated" and math.isclose(self.lambda_x, 0.0118):
            object.__setattr__(self, "lambda_x", 0.0212)

        # Validate p_bar in [ccr_min, ccr_max]
        if not (self.ccr_min <= self.p_bar <= self.ccr_max):
            raise ValueError(
                f"p_bar={self.p_bar} must be in [{self.ccr_min}, {self.ccr_max}]."
            )

        # Edge case 8: Validate alpha range
        # Paper requires alpha <= (ccr_max - p_bar) / (ccr_max - ccr_min)
        max_alpha = (self.ccr_max - self.p_bar) / (self.ccr_max - self.ccr_min)
        if not (0.0 <= self.alpha <= max_alpha + 1e-12):
            raise ValueError(
                f"Invalid alpha={self.alpha}. Must satisfy 0.0 <= alpha <= "
                f"({self.ccr_max} - {self.p_bar})/({self.ccr_max} - {self.ccr_min}) = {max_alpha:.4f}."
            )

        # Validate discount factor gamma < 1.0
        if self.gamma >= 1.0 or self.gamma <= 0.0:
            raise ValueError(
                f"Discount factor gamma={self.gamma:.4f} must be strictly in (0, 1)."
            )

        # Validate pension bounds: P_min must be strictly greater than subsistence level F
        if self.P_min <= self.F:
            raise ValueError(
                f"P_min={self.P_min} must be strictly greater than subsistence floor F={self.F}."
            )

        if self.P_max <= self.P_min:
            raise ValueError(
                f"P_max={self.P_max} must be strictly greater than P_min={self.P_min}."
            )

    @property
    def r_plus_lambda(self) -> float:
        """Return the effective discount rate r + lambda_x."""
        return self.r + self.lambda_x

    @property
    def gamma(self) -> float:
        """Discount factor per period: gamma = exp(-(lambda_x + beta) * delta).

        Reference: Paper Section 5.3, eq. (5.33).
        """
        return math.exp(-(self.lambda_x + self.beta) * self.delta)

    @property
    def V_min(self) -> float:
        """Lower wealth bound on grid."""
        return self.V_min_factor * self.V0

    @property
    def V_max(self) -> float:
        """Upper wealth bound on grid."""
        return self.V_max_factor * self.V0

    def utility(self, P: float | np.ndarray) -> float | np.ndarray:
        """Compute HARA utility U(P) = a_hat * ((1-b)/b) * ((P-F)/(1-b))^b.

        For b = -1, a_hat = 1: U(P) = -4.0 / (P - F).
        Reference: Paper eq. (5.34).
        """
        P_arr = np.asarray(P, dtype=np.float64)
        if np.any(P_arr <= self.F):
            # Clamp to prevent zero division / invalid values
            P_arr = np.maximum(P_arr, self.P_min)

        if math.isclose(self.b, -1.0) and math.isclose(self.a_hat, 1.0):
            res = -4.0 / (P_arr - self.F)
        else:
            base = (P_arr - self.F) / (1.0 - self.b)
            res = self.a_hat * ((1.0 - self.b) / self.b) * (base**self.b)

        return float(res) if np.ndim(P) == 0 else res

    def reward(self, P: float | np.ndarray) -> float | np.ndarray:
        """Compute one-period integrated discounted reward r(P).

        r(P) = [1 / (lambda_x + beta)] * (1 - exp(-(lambda_x + beta) * delta)) * U(P).
        Reference: Paper eq. (5.33), (5.35).
        """
        u = self.utility(P)
        denom = self.lambda_x + self.beta
        factor = (1.0 - math.exp(-denom * self.delta)) / denom
        return factor * u

    def initial_pension(self, alpha: float | None = None) -> float:
        """Compute initial pension P0(alpha).

        P0 = ((1 - alpha) / (p_bar - alpha)) * (r + lambda_x) * V0.
        Reference: Paper eq. (5.12) evaluated at initial state with CCR0 = (p_bar - alpha) / (1 - alpha).
        """
        a = self.alpha if alpha is None else alpha
        return ((1.0 - a) / (self.p_bar - a)) * self.r_plus_lambda * self.V0


def get_default_config(mode: ConfigMode = "paper_calibrated", alpha: float = 0.0) -> Config:
    """Convenience helper to create a validated Config instance."""
    return Config(mode=mode, alpha=alpha)

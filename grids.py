"""Discretized state, action, and shock grids for the pension MDP.

Implements grid construction and node mapping as described in:
Lichtenstern & Zagst (2022) - Section 5.3.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

from config import Config


@dataclass
class Grids:
    """Container for state, action, and shock discretization grids.

    Attributes:
        config: Configuration object.
        V_grid: Equidistant wealth grid [V_min, V_max] of length n_V.
        CCR_grid: Equidistant CCR grid [ccr_min, ccr_max] of length n_CCR.
        V_flat: Flat array of wealth values for each state index s in [0, n_S-1].
        CCR_flat: Flat array of CCR values for each state index s in [0, n_S-1].
        P_flat: Flat array of pension payments P(V, CCR) for each state index.
        reward_flat: One-period rewards r(P) precomputed for each state index.
        actions: Discrete action choices in [0.0, 1.0] representing risky asset share a.
        Z: Discrete shock values approximating standard normal distribution N(0, 1).
        Z_weights: Equal probability weights (1 / n_Z) for each shock node.
    """

    config: Config
    V_grid: np.ndarray
    CCR_grid: np.ndarray
    V_flat: np.ndarray
    CCR_flat: np.ndarray
    P_flat: np.ndarray
    reward_flat: np.ndarray
    actions: np.ndarray
    Z: np.ndarray
    Z_weights: np.ndarray

    @property
    def n_V(self) -> int:
        return len(self.V_grid)

    @property
    def n_CCR(self) -> int:
        return len(self.CCR_grid)

    @property
    def n_S(self) -> int:
        return len(self.V_flat)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def n_Z(self) -> int:
        return len(self.Z)

    def state_to_index(self, i: int | np.ndarray, j: int | np.ndarray) -> int | np.ndarray:
        """Map wealth index i and CCR index j to flat state index s = i * n_CCR + j."""
        return i * self.n_CCR + j

    def index_to_state(self, s: int | np.ndarray) -> tuple[int | np.ndarray, int | np.ndarray]:
        """Map flat state index s back to (wealth_index i, ccr_index j)."""
        return np.divmod(s, self.n_CCR)

    def snap_to_nearest_node(
        self, V: float | np.ndarray, CCR: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Snap arbitrary (V, CCR) coordinates to the nearest grid node.

        Edge Case 2: Nearest V index and nearest CCR index are computed separately.
        V is clamped to [V_min, V_max], CCR is clamped to [ccr_min, ccr_max].
        Nearest indices are determined by rounding the relative continuous coordinates.

        Returns:
            Tuple of (nearest_V_index, nearest_CCR_index, flat_state_index).
        """
        V_arr = np.asarray(V, dtype=np.float64)
        CCR_arr = np.asarray(CCR, dtype=np.float64)

        # Clamp continuous coordinates to grid boundaries
        V_clamped = np.clip(V_arr, self.config.V_min, self.config.V_max)
        CCR_clamped = np.clip(CCR_arr, self.config.ccr_min, self.config.ccr_max)

        # Relative coordinate scaling
        i_cont = (V_clamped - self.config.V_min) / (self.config.V_max - self.config.V_min) * (self.n_V - 1)
        j_cont = (CCR_clamped - self.config.ccr_min) / (self.config.ccr_max - self.config.ccr_min) * (self.n_CCR - 1)

        i_idx = np.clip(np.rint(i_cont).astype(np.int32), 0, self.n_V - 1)
        j_idx = np.clip(np.rint(j_cont).astype(np.int32), 0, self.n_CCR - 1)
        s_idx = self.state_to_index(i_idx, j_idx)

        return i_idx, j_idx, s_idx


def create_grids(config: Config) -> Grids:
    """Build the discretized state, action, and shock grids from configuration.

    Shock Grid Construction (Paper Section 5.3):
    - Quantile step q = 0.025, N_q = 39, n_Z = 40.
    - Interval midpoints: (k + 0.5) / 40 for k in 0..39.
    - z_k = norm.ppf((k + 0.5) / 40).
    - First value: norm.ppf(0.5 / 40) = norm.ppf(0.0125) approx -2.2414.
    - Last value: norm.ppf(39.5 / 40) = norm.ppf(0.9875) approx +2.2414.
    - Each shock node has equal probability 1 / 40 = 0.025.
    """
    # Wealth grid: equidistant points
    V_grid = np.linspace(config.V_min, config.V_max, config.n_V, dtype=np.float64)

    # CCR grid: equidistant points from 1.00 to 1.25 (step 0.01 when n_CCR=26)
    CCR_grid = np.linspace(config.ccr_min, config.ccr_max, config.n_CCR, dtype=np.float64)

    # Flat state representations: index s = i * n_CCR + j
    V_flat = np.repeat(V_grid, config.n_CCR)
    CCR_flat = np.tile(CCR_grid, config.n_V)

    # Pension for state (V_i, CCR_j): P = V_i * (r + lambda_x) / CCR_j
    # Reference: Paper eq. (5.9)
    P_flat = V_flat * config.r_plus_lambda / CCR_flat

    # Precomputed one-period reward for each state
    reward_flat = config.reward(P_flat)

    # Action grid: risky asset weights a in [0.0, 1.0] (no short-selling, no leverage)
    actions = np.linspace(0.0, 1.0, config.n_actions, dtype=np.float64)

    # Shock grid Z: norm.ppf of quantile midpoints
    k = np.arange(config.n_Z, dtype=np.float64)
    midpoints = (k + 0.5) / float(config.n_Z)
    Z = stats.norm.ppf(midpoints).astype(np.float64)
    Z_weights = np.full(config.n_Z, 1.0 / float(config.n_Z), dtype=np.float64)

    return Grids(
        config=config,
        V_grid=V_grid,
        CCR_grid=CCR_grid,
        V_flat=V_flat,
        CCR_flat=CCR_flat,
        P_flat=P_flat,
        reward_flat=reward_flat,
        actions=actions,
        Z=Z,
        Z_weights=Z_weights,
    )

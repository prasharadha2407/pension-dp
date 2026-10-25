"""Transition mechanics and grid precomputation for the pension MDP.

Vectorized transition function mapping (state, action, shock) -> next_state.
Implements equations (5.4), (5.5), (5.12), (5.19) from:
Lichtenstern & Zagst (2022).
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

from config import Config
from grids import Grids

logger = logging.getLogger(__name__)


class TransitionResult(NamedTuple):
    """Result of precomputing discrete MDP transitions.

    Attributes:
        next_state: Array of shape (n_S, n_actions, n_Z) with int32 next state indices.
        negative_wealth_count: Number of transitions where raw V_next was <= 0.
        v_min_clamped_count: Number of transitions where V_next fell below V_min.
        v_max_clamped_count: Number of transitions where V_next exceeded V_max.
    """

    next_state: np.ndarray
    negative_wealth_count: int
    v_min_clamped_count: int
    v_max_clamped_count: int


def continuous_transition_step(
    V: np.ndarray | float,
    P: np.ndarray | float,
    a: np.ndarray | float,
    Z: np.ndarray | float,
    config: Config,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Perform one continuous time-step transition (V, P) -> (V_next, P_next, CCR_next).

    Mathematical Formulation:
    - Mathematical reserve: E = P / (r + lambda_x) (Paper eq. 5.4)
    - Invested wealth: invested = E + (1 - alpha) * (V - E) (Paper eq. 5.4, 5.5)
    - Wealth dynamic (Euler step over delta):
      V_next = V + invested * [(r + a*(mu - r))*delta + a*sigma*sqrt(delta)*Z] - P*delta (Paper eq. 5.19)
    - Regulation check:
      ratio = V_next / E
      If ccr_min - tol <= ratio <= ccr_max + tol:
          P_next = P  (Inside corridor: no regulatory adjustment)
      Else:
          P_next = ((1 - alpha) / (p_bar - alpha)) * (r + lambda_x) * V_next (Paper eq. 5.12)
    - Clamping:
      V_next is clamped to [V_min, V_max] (Edge case 1, 4).
      P_next is clamped to [P_min, P_max] (Edge case 3).
      CCR_next = V_next * (r + lambda_x) / P_next, clamped to [ccr_min, ccr_max].
    """
    V_arr = np.asarray(V, dtype=np.float64)
    P_arr = np.asarray(P, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    Z_arr = np.asarray(Z, dtype=np.float64)

    r_plus_lambda = config.r_plus_lambda
    delta = config.delta
    sqrt_delta = math.sqrt(delta)

    # 1. Mathematical reserve E (Paper eq. 5.4)
    E = P_arr / r_plus_lambda

    # 2. Invested wealth in asset portfolio vs buffer portfolio (Paper eq. 5.4, 5.5)
    invested = E + (1.0 - config.alpha) * (V_arr - E)

    # 3. Wealth return and update (Paper eq. 5.19)
    drift = (config.r + a_arr * (config.mu - config.r)) * delta
    diffusion = a_arr * config.sigma * sqrt_delta * Z_arr
    V_next = V_arr + invested * (drift + diffusion) - P_arr * delta

    # Clamp wealth to avoid negative/out-of-bounds values
    V_clamped = np.clip(V_next, config.V_min, config.V_max)

    # 4. Corridor ratio and regulation check (Paper eq. 5.9, 5.12)
    ratio = V_next / np.maximum(E, 1e-12)
    tol = config.corridor_tolerance
    inside_corridor = (ratio >= config.ccr_min - tol) & (ratio <= config.ccr_max + tol)

    # Regulatory pension reset formula (Paper eq. 5.12)
    reset_factor = ((1.0 - config.alpha) / (config.p_bar - config.alpha)) * r_plus_lambda
    P_reset = np.clip(reset_factor * V_clamped, config.P_min, config.P_max)

    P_next = np.where(inside_corridor, P_arr, P_reset)
    P_next = np.clip(P_next, config.P_min, config.P_max)

    # 5. CCR next
    CCR_next = np.clip(V_clamped * r_plus_lambda / P_next, config.ccr_min, config.ccr_max)

    if np.ndim(V) == 0:
        return float(V_clamped), float(P_next), float(CCR_next)
    return V_clamped, P_next, CCR_next


def precompute_transitions(
    grids: Grids,
    chunk_size: int = 2600,
) -> TransitionResult:
    """Precompute next_state[s, a, z] index mapping as an int32 array.

    Vectorized across state chunks to ensure high performance and minimal memory footprint.
    Handles all boundary clamps and flags occurrences of zero/negative wealth.

    Returns:
        TransitionResult containing the int32 next_state tensor and diagnostics.
    """
    config = grids.config
    n_S = grids.n_S
    n_a = grids.n_actions
    n_Z = grids.n_Z

    next_state = np.empty((n_S, n_a, n_Z), dtype=np.int32)

    r_plus_lambda = config.r_plus_lambda
    delta = config.delta
    sqrt_delta = math.sqrt(delta)
    V_min = config.V_min
    V_max = config.V_max
    ccr_min = config.ccr_min
    ccr_max = config.ccr_max
    n_V = grids.n_V
    n_CCR = grids.n_CCR
    tol = config.corridor_tolerance

    # Reshape action and shock arrays for broadcasting:
    # a_arr shape: (1, n_a, 1)
    # Z_arr shape: (1, 1, n_Z)
    a_arr = grids.actions[np.newaxis, :, np.newaxis]
    Z_arr = grids.Z[np.newaxis, np.newaxis, :]

    # Precompute return factor matrix of shape (1, n_a, n_Z)
    drift = (config.r + a_arr * (config.mu - config.r)) * delta
    diffusion = a_arr * config.sigma * sqrt_delta * Z_arr
    return_factor = drift + diffusion

    reset_factor = ((1.0 - config.alpha) / (config.p_bar - config.alpha)) * r_plus_lambda

    neg_wealth_total = 0
    v_min_clamp_total = 0
    v_max_clamp_total = 0

    # Process states in chunks
    for start_idx in range(0, n_S, chunk_size):
        end_idx = min(start_idx + chunk_size, n_S)

        # Slice flat state arrays: shape (B, 1, 1)
        V_chunk = grids.V_flat[start_idx:end_idx, np.newaxis, np.newaxis]
        P_chunk = grids.P_flat[start_idx:end_idx, np.newaxis, np.newaxis]

        # Reserve E and invested wealth (Paper eq. 5.4, 5.5)
        E_chunk = P_chunk / r_plus_lambda
        invested_chunk = E_chunk + (1.0 - config.alpha) * (V_chunk - E_chunk)

        # Wealth update (Paper eq. 5.19)
        V_next = V_chunk + invested_chunk * return_factor - P_chunk * delta

        # Diagnostics: track raw boundary violations
        neg_wealth_total += int(np.sum(V_next <= 0.0))
        v_min_clamp_total += int(np.sum(V_next < V_min))
        v_max_clamp_total += int(np.sum(V_next > V_max))

        # Clamp wealth (Edge case 1, 4)
        V_clamped = np.clip(V_next, V_min, V_max)

        # Corridor test: ratio = V_next / E (Paper eq. 5.9)
        ratio = V_next / E_chunk
        inside_corridor = (ratio >= ccr_min - tol) & (ratio <= ccr_max + tol)

        # Pension reset when outside corridor (Paper eq. 5.12)
        P_reset = np.clip(reset_factor * V_clamped, config.P_min, config.P_max)
        P_next = np.where(inside_corridor, P_chunk, P_reset)
        P_next = np.clip(P_next, config.P_min, config.P_max)

        # Snapping to nearest grid node (Edge case 2)
        # Separate nearest V index and nearest CCR index
        CCR_next = np.clip(V_clamped * r_plus_lambda / P_next, ccr_min, ccr_max)

        # Continuous relative indices
        i_cont = (V_clamped - V_min) / (V_max - V_min) * (n_V - 1)
        j_cont = (CCR_next - ccr_min) / (ccr_max - ccr_min) * (n_CCR - 1)

        i_idx = np.clip(np.rint(i_cont).astype(np.int32), 0, n_V - 1)
        j_idx = np.clip(np.rint(j_cont).astype(np.int32), 0, n_CCR - 1)

        # State index s' = i * n_CCR + j
        next_state[start_idx:end_idx, :, :] = i_idx * n_CCR + j_idx

    logger.info(
        "Transitions precomputed for %d states: %d negative wealth events, "
        "%d clamped to V_min, %d clamped to V_max.",
        n_S,
        neg_wealth_total,
        v_min_clamp_total,
        v_max_clamp_total,
    )

    return TransitionResult(
        next_state=next_state,
        negative_wealth_count=neg_wealth_total,
        v_min_clamped_count=v_min_clamp_total,
        v_max_clamped_count=v_max_clamp_total,
    )

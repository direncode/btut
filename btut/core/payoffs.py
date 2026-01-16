"""
Payoff matrices for game-theoretic coordination.

This module implements various two-player game payoff structures used
in BTUT coordination. The default is a mixed Stag Hunt + Prisoner's Dilemma
game that captures both coordination and cooperation incentives.

Theory:
    In BTUT, agents play a composite game:
    - Stag Hunt (SH): Coordination game where mutual cooperation yields highest payoff
    - Prisoner's Dilemma (PD): Social dilemma where defection dominates individually

    The mixing parameter α ∈ [0,1] blends these:
        U = α · U_SH + (1-α) · U_PD

    The cooperation bonus γ scales cooperative payoffs:
        U_A(cooperate) = γ · base_payoff

    Phase transition occurs at critical γ_c ≈ 1.33 where the system
    switches from all-defect to all-cooperate equilibrium.

Example:
    >>> payoff = StagHuntPD(gamma=1.5, alpha=0.6)
    >>> u_coop, u_defect = payoff.expected_utilities(cooperation_rate=0.7)
    >>> print(f"Cooperate: {u_coop:.3f}, Defect: {u_defect:.3f}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


class PayoffMatrix(ABC):
    """
    Abstract base class for game-theoretic payoff matrices.

    Subclasses must implement `expected_utilities` to compute
    expected payoffs given the current cooperation rate.

    This abstraction allows users to plug in custom games
    (Chicken, Snowdrift, etc.) while maintaining the BTUT framework.
    """

    @abstractmethod
    def expected_utilities(
        self,
        cooperation_rate: float,
        hub_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute expected utilities for cooperate vs defect.

        Args:
            cooperation_rate: Fraction of population cooperating (p).
            hub_rate: Optional separate cooperation rate for hubs.

        Returns:
            Tuple of (U_cooperate, U_defect).
        """
        pass

    @abstractmethod
    def payoff_matrix(self) -> np.ndarray:
        """
        Return the 2x2 payoff matrix.

        Returns:
            2x2 numpy array where [i,j] = payoff to row player
            when row plays i and column plays j.
            Convention: 0 = cooperate, 1 = defect.
        """
        pass


@dataclass
class PrisonersDilemma(PayoffMatrix):
    """
    Classic Prisoner's Dilemma payoff structure.

    Payoff matrix:
                    Cooperate   Defect
        Cooperate      R          S
        Defect         T          P

    Where T > R > P > S (temptation > reward > punishment > sucker).

    Attributes:
        R: Reward for mutual cooperation (default: 3).
        S: Sucker's payoff (cooperate vs defector) (default: 0).
        T: Temptation to defect (default: 5).
        P: Punishment for mutual defection (default: 1).
        gamma: Cooperation bonus multiplier applied to R and S.

    Example:
        >>> pd = PrisonersDilemma(gamma=1.5)
        >>> u_c, u_d = pd.expected_utilities(0.5)
    """

    R: float = 3.0  # Reward
    S: float = 0.0  # Sucker
    T: float = 5.0  # Temptation
    P: float = 1.0  # Punishment
    gamma: float = 1.0

    def __post_init__(self) -> None:
        """Validate PD constraints: T > R > P > S."""
        if not (self.T > self.R > self.P > self.S):
            raise ValueError(
                f"PD requires T > R > P > S, got T={self.T}, R={self.R}, "
                f"P={self.P}, S={self.S}"
            )

    def payoff_matrix(self) -> np.ndarray:
        """Return 2x2 PD payoff matrix."""
        return np.array([
            [self.gamma * self.R, self.gamma * self.S],
            [self.T, self.P]
        ])

    def expected_utilities(
        self,
        cooperation_rate: float,
        hub_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute expected utilities in PD.

        E[U_cooperate] = p · γR + (1-p) · γS
        E[U_defect] = p · T + (1-p) · P
        """
        p = cooperation_rate

        u_cooperate = p * self.gamma * self.R + (1 - p) * self.gamma * self.S
        u_defect = p * self.T + (1 - p) * self.P

        return u_cooperate, u_defect


@dataclass
class StagHunt(PayoffMatrix):
    """
    Stag Hunt coordination game.

    Payoff matrix:
                    Cooperate   Defect
        Cooperate      A          C
        Defect         C          B

    Where A > B > C (mutual cooperation best, but defection is safe).

    Attributes:
        A: Payoff for mutual cooperation (hunting stag together).
        B: Payoff for mutual defection (both hunt hare).
        C: Payoff for miscoordination (one hunts stag, other hunts hare).
        gamma: Cooperation bonus multiplier.

    Example:
        >>> sh = StagHunt(A=4.0, B=2.0, C=1.0, gamma=1.5)
        >>> u_c, u_d = sh.expected_utilities(0.7)
    """

    A: float = 4.0  # Mutual cooperation
    B: float = 2.0  # Mutual defection
    C: float = 1.0  # Miscoordination
    gamma: float = 1.0

    def __post_init__(self) -> None:
        """Validate SH constraints: A > B > C."""
        if not (self.A > self.B >= self.C):
            raise ValueError(
                f"SH requires A > B >= C, got A={self.A}, B={self.B}, C={self.C}"
            )

    def payoff_matrix(self) -> np.ndarray:
        """Return 2x2 SH payoff matrix."""
        return np.array([
            [self.gamma * self.A, self.C],
            [self.C, self.B]
        ])

    def expected_utilities(
        self,
        cooperation_rate: float,
        hub_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute expected utilities in Stag Hunt.

        E[U_cooperate] = p · γA + (1-p) · C
        E[U_defect] = p · C + (1-p) · B
        """
        p = cooperation_rate

        u_cooperate = p * self.gamma * self.A + (1 - p) * self.C
        u_defect = p * self.C + (1 - p) * self.B

        return u_cooperate, u_defect


@dataclass
class StagHuntPD(PayoffMatrix):
    """
    Mixed Stag Hunt + Prisoner's Dilemma payoff structure.

    This is the default payoff matrix for BTUT, combining coordination
    incentives (SH) with social dilemma structure (PD).

    The mixing is controlled by parameter α:
        U = α · U_SH + (1-α) · U_PD

    Attributes:
        gamma: Cooperation bonus multiplier (critical point γ_c ≈ 1.33).
        alpha: Mixing weight for Stag Hunt (0 = pure PD, 1 = pure SH).
        cA_SH: Cost of cooperation in Stag Hunt.
        cB_SH: Cost of defection in Stag Hunt.
        cA_PD: Cost of cooperation in Prisoner's Dilemma.
        cB_PD: Cost of defection in Prisoner's Dilemma.

    Theory:
        The expected utility for strategy A (cooperate) given cooperation rate p:

        U_A = α · [p·γ·(1-cA_SH) + (1-p)·(-cA_SH)] +
              (1-α) · [p·γ·R + (1-p)·γ·S - cA_PD]

        Phase transition occurs when U_A = U_B, which defines γ_c.

    Example:
        >>> payoff = StagHuntPD(gamma=1.5, alpha=0.6)
        >>> u_c, u_d = payoff.expected_utilities(cooperation_rate=0.7)
        >>> print(f"Advantage to cooperate: {u_c - u_d:.3f}")
    """

    gamma: float = 1.45
    alpha: float = 0.6
    cA_SH: float = 0.40
    cB_SH: float = 0.10
    cA_PD: float = 0.20
    cB_PD: float = 0.08

    # Internal PD parameters (scaled)
    _R: float = 3.0
    _S: float = 0.0
    _T: float = 5.0
    _P: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.gamma < 0:
            raise ValueError(f"Gamma must be non-negative, got {self.gamma}")
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")

    def payoff_matrix(self) -> np.ndarray:
        """
        Return effective 2x2 payoff matrix.

        This is a linearization around p=0.5 for visualization;
        actual payoffs depend on cooperation rate.
        """
        # Approximate at p = 0.5
        u_cc = self.alpha * self.gamma * (1 - self.cA_SH) + \
               (1 - self.alpha) * self.gamma * self._R
        u_cd = self.alpha * (-self.cA_SH) + \
               (1 - self.alpha) * self.gamma * self._S
        u_dc = self.alpha * (1 - self.cB_SH) + \
               (1 - self.alpha) * self._T
        u_dd = self.alpha * (1 - self.cB_SH) + \
               (1 - self.alpha) * self._P

        return np.array([[u_cc, u_cd], [u_dc, u_dd]])

    def expected_utilities(
        self,
        cooperation_rate: float,
        hub_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute expected utilities for cooperate vs defect.

        This implements the core BTUT payoff calculation that determines
        strategy adoption dynamics.

        Args:
            cooperation_rate: Fraction p of population cooperating.
            hub_rate: Optional separate hub cooperation rate (for τ > 0).

        Returns:
            Tuple of (U_cooperate, U_defect).

        Mathematical derivation:
            Let p = cooperation_rate, γ = gamma, α = alpha.

            Stag Hunt component:
                EU_SH_A = 0.5 · (u_A + p·u_A + (1-p)·u_B) · (p·γ + (1-p)·ε) - cA_SH·u_A
                EU_SH_B = 0.5 · (u_B + p·u_A + (1-p)·u_B) · (p·ε + (1-p)·1) - cB_SH·u_B

            Prisoner's Dilemma component:
                EU_PD_A = p·γ·R + (1-p)·γ·S - cA_PD
                EU_PD_B = p·T + (1-p)·P - cB_PD

            Combined:
                U_A = α · EU_SH_A + (1-α) · EU_PD_A
                U_B = α · EU_SH_B + (1-α) · EU_PD_B
        """
        p = cooperation_rate
        gamma = self.gamma
        alpha = self.alpha

        # Base utility values (log-scaled for numerical stability)
        u_A = np.log1p(1.0)  # ~0.693
        u_B = np.log1p(0.5)  # ~0.405

        # Small epsilon for defection bonus
        eps = 0.03

        # === Stag Hunt Component ===
        # Expected utility when cooperating in SH
        coordination_bonus = 0.5 * (u_A + p * u_A + (1 - p) * u_B)
        cooperation_multiplier = p * gamma + (1 - p) * eps
        EU_SH_A = coordination_bonus * cooperation_multiplier - self.cA_SH * u_A

        # Expected utility when defecting in SH
        defection_base = 0.5 * (u_B + p * u_A + (1 - p) * u_B)
        defection_multiplier = p * eps + (1 - p) * 1.0
        EU_SH_B = defection_base * defection_multiplier - self.cB_SH * u_B

        # === Prisoner's Dilemma Component ===
        EU_PD_A = p * gamma * self._R + (1 - p) * gamma * self._S - self.cA_PD
        EU_PD_B = p * self._T + (1 - p) * self._P - self.cB_PD

        # === Hawk-Dove Component (small mixing for stability) ===
        # This adds a small anti-coordination component
        EU_HD_A = 0.1 * (1 - p)  # Slight advantage when others defect
        EU_HD_B = 0.1 * p  # Slight advantage when others cooperate

        # === Combined Utility ===
        mix = alpha
        U_A = mix * (EU_SH_A + EU_HD_A) + (1 - mix) * EU_PD_A
        U_B = mix * (EU_SH_B + EU_HD_B) + (1 - mix) * EU_PD_B

        return float(U_A), float(U_B)

    def find_critical_gamma(
        self,
        tolerance: float = 1e-4,
        max_iterations: int = 100,
    ) -> float:
        """
        Find critical gamma where U_A = U_B at p = 0.5.

        This is a simplified analytical estimate of γ_c.
        For precise values, use CriticalPointFinder.

        Args:
            tolerance: Convergence tolerance.
            max_iterations: Maximum binary search iterations.

        Returns:
            Estimated critical gamma value.
        """
        low, high = 1.0, 2.0

        for _ in range(max_iterations):
            mid = (low + high) / 2
            old_gamma = self.gamma
            self.gamma = mid

            u_a, u_b = self.expected_utilities(0.5)
            self.gamma = old_gamma

            if abs(u_a - u_b) < tolerance:
                return mid
            elif u_a > u_b:
                high = mid
            else:
                low = mid

        return (low + high) / 2

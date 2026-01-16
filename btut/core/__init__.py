"""
btut.core - Core primitives for BTUT coordination
=================================================

Contains:
    - Agent: Base agent class with strategy and utility tracking
    - PayoffMatrix: Abstract base for game-theoretic payoffs
    - StagHuntPD: Mixed Stag Hunt + Prisoner's Dilemma payoff
    - MeanFieldDynamics: O(N) update rule implementation
"""

from btut.core.agent import Agent, Strategy
from btut.core.payoffs import PayoffMatrix, StagHuntPD, PrisonersDilemma, StagHunt
from btut.core.dynamics import MeanFieldDynamics
from btut.core.presets import PRESETS

__all__ = [
    "Agent",
    "Strategy",
    "PayoffMatrix",
    "StagHuntPD",
    "PrisonersDilemma",
    "StagHunt",
    "MeanFieldDynamics",
    "PRESETS",
]

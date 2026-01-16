"""
Unit tests for btut.core module.

Tests cover:
- Agent creation and properties
- Payoff matrix calculations
- Mean-field dynamics updates
"""

import pytest
import numpy as np

from btut.core.agent import Agent, Strategy, AgentPopulation
from btut.core.payoffs import PayoffMatrix, StagHuntPD, PrisonersDilemma, StagHunt
from btut.core.dynamics import MeanFieldDynamics, ConvergenceChecker
from btut.core.presets import PRESETS, get_preset


class TestAgent:
    """Tests for Agent class."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent(id=0, degree=5, tau=0.3)
        assert agent.id == 0
        assert agent.degree == 5
        assert agent.tau == 0.3
        assert agent.strategy == Strategy.DEFECT  # Default

    def test_agent_weight(self):
        """Test hub weight calculation."""
        agent = Agent(id=0, degree=10, tau=0.5)
        # weight = degree^tau = 10^0.5 ≈ 3.16
        assert abs(agent.weight - 10**0.5) < 1e-10

    def test_agent_weight_tau_zero(self):
        """Weight should be 1 when tau=0."""
        agent = Agent(id=0, degree=100, tau=0.0)
        assert agent.weight == 1.0

    def test_agent_weight_tau_one(self):
        """Weight should equal degree when tau=1."""
        agent = Agent(id=0, degree=10, tau=1.0)
        assert agent.weight == 10.0

    def test_agent_switch_strategy(self):
        """Test strategy switching."""
        agent = Agent(id=0, degree=5, tau=0.3)
        assert agent.strategy == Strategy.DEFECT
        agent.switch_strategy()
        assert agent.strategy == Strategy.COOPERATE
        agent.switch_strategy()
        assert agent.strategy == Strategy.DEFECT

    def test_agent_invalid_degree(self):
        """Should raise error for invalid degree."""
        with pytest.raises(ValueError):
            Agent(id=0, degree=0, tau=0.3)

    def test_agent_invalid_tau(self):
        """Should raise error for tau outside [0,1]."""
        with pytest.raises(ValueError):
            Agent(id=0, degree=5, tau=1.5)

    def test_agent_to_dict(self):
        """Test dictionary conversion."""
        agent = Agent(id=0, degree=5, tau=0.3)
        d = agent.to_dict()
        assert d["id"] == 0
        assert d["degree"] == 5
        assert "weight" in d


class TestAgentPopulation:
    """Tests for AgentPopulation class."""

    def test_population_creation(self):
        """Test population creation from degrees."""
        degrees = np.array([1, 2, 3, 4, 5])
        pop = AgentPopulation.from_degrees(degrees, tau=0.3, seed=42)
        assert pop.n_agents == 5
        assert len(pop.strategies) == 5

    def test_cooperation_rate(self):
        """Test cooperation rate calculation."""
        pop = AgentPopulation(
            n_agents=4,
            degrees=np.array([1, 2, 3, 4]),
            tau=0.3,
            strategies=np.array([True, True, False, False]),
            utilities=np.zeros(4),
        )
        assert pop.cooperation_rate == 0.5

    def test_weighted_cooperation(self):
        """Test weighted cooperation calculation."""
        pop = AgentPopulation(
            n_agents=2,
            degrees=np.array([1, 10]),  # Low and high degree
            tau=1.0,  # Weight = degree
            strategies=np.array([True, False]),  # Only low-degree cooperates
            utilities=np.zeros(2),
        )
        # Weighted: (1*1 + 10*0) / (1 + 10) = 1/11 ≈ 0.091
        assert abs(pop.weighted_cooperation - 1/11) < 1e-10


class TestStagHuntPD:
    """Tests for StagHuntPD payoff matrix."""

    def test_payoff_creation(self):
        """Test payoff matrix creation."""
        payoff = StagHuntPD(gamma=1.5, alpha=0.6)
        assert payoff.gamma == 1.5
        assert payoff.alpha == 0.6

    def test_expected_utilities(self):
        """Test utility calculation."""
        payoff = StagHuntPD(gamma=1.5, alpha=0.6)
        u_coop, u_defect = payoff.expected_utilities(0.5)
        # Just check they're finite numbers
        assert np.isfinite(u_coop)
        assert np.isfinite(u_defect)

    def test_cooperation_advantage_high_gamma(self):
        """High gamma should favor cooperation."""
        payoff = StagHuntPD(gamma=2.0, alpha=0.6)
        u_coop, u_defect = payoff.expected_utilities(0.5)
        # At high gamma, cooperation should be advantageous
        assert u_coop > u_defect

    def test_defection_advantage_low_gamma(self):
        """Low gamma should favor defection."""
        payoff = StagHuntPD(gamma=1.0, alpha=0.6)
        u_coop, u_defect = payoff.expected_utilities(0.5)
        # At low gamma, defection should be advantageous
        assert u_defect > u_coop


class TestMeanFieldDynamics:
    """Tests for MeanFieldDynamics."""

    def test_compute_field(self):
        """Test mean-field computation."""
        pop = AgentPopulation.from_degrees(
            np.array([1, 2, 3, 4, 5]),
            tau=0.3,
            initial_cooperation=0.5,
            seed=42,
        )
        payoff = StagHuntPD(gamma=1.5)
        dynamics = MeanFieldDynamics(payoff=payoff)

        p, p_hub = dynamics.compute_field(pop)
        assert 0 <= p <= 1
        assert 0 <= p_hub <= 1

    def test_update_preserves_population_size(self):
        """Update should not change population size."""
        pop = AgentPopulation.from_degrees(
            np.ones(100, dtype=np.int32) * 3,
            tau=0.3,
            seed=42,
        )
        payoff = StagHuntPD(gamma=1.5)
        dynamics = MeanFieldDynamics(payoff=payoff)

        new_pop = dynamics.update(pop)
        assert new_pop.n_agents == pop.n_agents


class TestConvergenceChecker:
    """Tests for ConvergenceChecker."""

    def test_not_converged_early(self):
        """Should not report convergence before window filled."""
        checker = ConvergenceChecker(window=5, tolerance=1e-6)
        for _ in range(3):
            assert not checker.update(0.5)

    def test_converged_stable(self):
        """Should detect convergence when values stable."""
        checker = ConvergenceChecker(window=3, tolerance=1e-6)
        checker.update(0.9)
        checker.update(0.9)
        assert checker.update(0.9)  # Third value -> converged

    def test_not_converged_varying(self):
        """Should not report convergence when values vary."""
        checker = ConvergenceChecker(window=3, tolerance=0.01)
        checker.update(0.5)
        checker.update(0.7)
        assert not checker.update(0.9)


class TestPresets:
    """Tests for presets."""

    def test_all_presets_exist(self):
        """All expected presets should exist."""
        expected = ['quick', 'standard', 'large', 'high_cooperation']
        for name in expected:
            assert name in PRESETS

    def test_get_preset(self):
        """get_preset should return valid config."""
        config = get_preset('standard')
        assert 'n_agents' in config
        assert 'gamma' in config
        assert 'tau' in config

    def test_invalid_preset(self):
        """Should raise KeyError for invalid preset."""
        with pytest.raises(KeyError):
            get_preset('nonexistent')

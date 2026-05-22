"""Tests for configuration."""

from genesis_v2.config import GenesisConfig, PhysicsConfig, load_config


class TestConfig:
    def test_default_physics(self):
        phy = PhysicsConfig()
        assert phy.alpha == 0.01
        assert phy.w_pred == 1.0
        assert phy.initial_energy == 5000.0

    def test_load_config_default(self):
        cfg = load_config()
        assert isinstance(cfg, GenesisConfig)
        assert cfg.physics.alpha == 0.01

    def test_genesis_config_defaults(self):
        cfg = GenesisConfig()
        assert cfg.genome.node_dim == 64
        assert cfg.genome.input_nodes == 8
        assert cfg.environment.type == "mock"

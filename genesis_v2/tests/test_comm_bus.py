"""Tests for CommunicationBus — grid topology messaging."""

import numpy as np
import pytest

from genesis_v2.genome.graph import D_MESSAGE
from genesis_v2.social.comm_bus import CommunicationBus


@pytest.fixture
def bus():
    return CommunicationBus(grid_rows=5, grid_cols=5, comm_radius=2, rng=np.random.default_rng(42))


class TestCommunicationBus:
    def test_assign_positions(self, bus):
        ids = [f"agent-{i}" for i in range(10)]
        bus.assign_positions(ids)
        assert len(bus.agent_positions) == 10
        for aid in ids:
            assert aid in bus.agent_positions
            assert aid in bus._inboxes

    def test_deliver_within_radius(self, bus):
        ids = ["a", "b", "c"]
        bus.assign_positions(ids)
        # Manually place agents close together
        bus.agent_positions["a"] = (2, 2)
        bus.agent_positions["b"] = (2, 3)  # dist=1, within radius=2
        bus.agent_positions["c"] = (0, 0)  # dist~2.8, outside radius=2

        msg = np.ones(D_MESSAGE, dtype=np.float32)
        recipients = bus.deliver("a", msg)
        assert recipients >= 1  # b should receive

        inbox_b = bus.get_inbox("b")
        assert np.allclose(inbox_b, msg)

        inbox_c = bus.get_inbox("c")
        assert np.allclose(inbox_c, 0.0)  # too far

    def test_deliver_no_self(self, bus):
        ids = ["a", "b"]
        bus.assign_positions(ids)
        bus.agent_positions["a"] = (2, 2)
        bus.agent_positions["b"] = (2, 2)  # same position

        msg = np.ones(D_MESSAGE, dtype=np.float32)
        bus.deliver("a", msg)
        inbox_a = bus.get_inbox("a")
        assert np.allclose(inbox_a, 0.0)  # sender doesn't receive own message

    def test_inbox_mean_pooling(self, bus):
        ids = ["a", "b"]
        bus.assign_positions(ids)
        bus.agent_positions["a"] = (2, 2)
        bus.agent_positions["b"] = (2, 3)

        msg1 = np.ones(D_MESSAGE, dtype=np.float32) * 2.0
        msg2 = np.ones(D_MESSAGE, dtype=np.float32) * 4.0
        bus.deliver("a", msg1)
        bus.deliver("a", msg2)  # same sender, two messages

        inbox_b = bus.get_inbox("b")
        np.testing.assert_allclose(inbox_b, 3.0)  # mean of 2 and 4

    def test_empty_inbox(self, bus):
        ids = ["a"]
        bus.assign_positions(ids)
        inbox = bus.get_inbox("a")
        assert inbox.shape == (D_MESSAGE,)
        assert np.allclose(inbox, 0.0)

    def test_clear_all(self, bus):
        ids = ["a", "b"]
        bus.assign_positions(ids)
        bus.agent_positions["a"] = (2, 2)
        bus.agent_positions["b"] = (2, 3)

        bus.deliver("a", np.ones(D_MESSAGE, dtype=np.float32))
        assert bus.message_count("b") == 1

        bus.clear_all()
        assert bus.message_count("b") == 0
        assert np.allclose(bus.get_inbox("b"), 0.0)

    def test_message_count(self, bus):
        ids = ["a", "b", "c"]
        bus.assign_positions(ids)
        bus.agent_positions["a"] = (2, 2)
        bus.agent_positions["b"] = (2, 3)
        bus.agent_positions["c"] = (2, 4)

        bus.deliver("a", np.ones(D_MESSAGE, dtype=np.float32))
        bus.deliver("c", np.ones(D_MESSAGE, dtype=np.float32))
        # b should receive from both a and c
        assert bus.message_count("b") == 2

    def test_inbox_raw(self, bus):
        ids = ["a", "b"]
        bus.assign_positions(ids)
        bus.agent_positions["a"] = (2, 2)
        bus.agent_positions["b"] = (2, 3)

        msg = np.arange(D_MESSAGE, dtype=np.float32)
        bus.deliver("a", msg)
        raw = bus.get_inbox_raw("b")
        assert len(raw) == 1
        np.testing.assert_array_equal(raw[0], msg)

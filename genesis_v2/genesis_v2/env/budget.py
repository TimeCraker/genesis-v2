"""BudgetManager — API cost tracking with auto-downgrade to Mock environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetManager:
    total_budget: float = 50.0
    per_island_budget: float = 15.0
    fallback_to_mock: bool = True

    spent_total: float = 0.0
    spent_per_island: dict[int, float] = field(default_factory=dict)

    def check_budget(self, island_id: int, estimated_cost: float) -> bool:
        """Return True if the API call is affordable."""
        if self.spent_total + estimated_cost > self.total_budget:
            return False
        island_spent = self.spent_per_island.get(island_id, 0.0)
        if island_spent + estimated_cost > self.per_island_budget:
            return False
        return True

    def record_cost(self, island_id: int, cost: float) -> None:
        """Record actual API cost."""
        self.spent_total += cost
        self.spent_per_island[island_id] = self.spent_per_island.get(island_id, 0.0) + cost

    def should_fallback(self, island_id: int) -> bool:
        """Check if island should downgrade to Mock environment."""
        if not self.fallback_to_mock:
            return False
        island_spent = self.spent_per_island.get(island_id, 0.0)
        return island_spent >= self.per_island_budget

    def get_cost_pressure(self, agent_id: str) -> float:
        """Return cost pressure as a fraction of remaining budget."""
        if self.total_budget <= 0:
            return 1.0
        return self.spent_total / self.total_budget

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.spent_total)

    def reset(self) -> None:
        self.spent_total = 0.0
        self.spent_per_island.clear()

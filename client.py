"""
Monte Carlo Tree Search (MCTS) Agent Reasoning and Action Planner.
Zero external dependencies, standard library only.
"""

import math
from typing import Dict, List, Any, Optional, Callable

class MCTSNode:
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0.0

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.q_value + exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)

class AgentMCTSPlannerClient:
    """
    Implements Monte Carlo Tree Search (MCTS) for multi-step agent reasoning:
    1. Selection via Upper Confidence Bound for Trees (UCB1)
    2. Expansion of unvisited actions
    3. Rollout evaluation
    4. Backpropagation of reward values
    """

    def __init__(self, exploration_constant: float = 1.414):
        self.c = exploration_constant

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Selects child with maximum UCB1 score."""
        return max(node.children, key=lambda c: c.ucb1(self.c))

    def backpropagate(self, node: MCTSNode, reward: float):
        """Propagates evaluation score upwards to root."""
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.total_value += reward
            curr = curr.parent

    def run_mcts(self, initial_state: str, possible_actions_fn: Callable[[str], List[str]], evaluate_state_fn: Callable[[str], float], iterations: int = 50) -> Dict[str, Any]:
        """
        Executes MCTS search and returns optimal sequence of actions.
        """
        root = MCTSNode(state=initial_state)

        for _ in range(iterations):
            node = root

            # 1. Selection
            while node.children:
                node = self.select_best_child(node)

            # 2. Expansion
            actions = possible_actions_fn(node.state)
            if actions and node.visits > 0:
                for act in actions:
                    next_state = f"{node.state} -> {act}"
                    child = MCTSNode(state=next_state, parent=node, action=act)
                    node.children.append(child)
                node = node.children[0]

            # 3. Rollout Evaluation
            reward = evaluate_state_fn(node.state)

            # 4. Backpropagation
            self.backpropagate(node, reward)

        # Select most visited child at root
        if not root.children:
            return {"best_action": None, "q_value": 0.0, "visits": 0}

        best_child = max(root.children, key=lambda c: c.visits)
        return {
            "best_action": best_child.action,
            "q_value": round(best_child.q_value, 3),
            "visits": best_child.visits,
            "total_candidates": len(root.children)
        }

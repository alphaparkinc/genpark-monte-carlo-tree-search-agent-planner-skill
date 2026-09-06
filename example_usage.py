"""
Demonstration of genpark-monte-carlo-tree-search-agent-planner-skill
"""

from client import AgentMCTSPlannerClient

def main():
    planner = AgentMCTSPlannerClient()

    # Define branch actions
    def candidate_actions(state: str):
        if "solve" in state:
            return []
        return ["decompose_subproblems", "retrieve_knowledge", "direct_inference"]

    # Reward heuristics
    def evaluate_state(state: str):
        if "decompose_subproblems" in state:
            return 0.95
        elif "retrieve_knowledge" in state:
            return 0.70
        return 0.30

    result = planner.run_mcts("start_problem", candidate_actions, evaluate_state, iterations=30)
    print("=== MCTS PLANNING REPORT ===")
    print("Best Recommended Action:", result["best_action"])
    print("Q-Value (Estimated Reward):", result["q_value"])
    print("Visit Count:", result["visits"])

if __name__ == "__main__":
    main()

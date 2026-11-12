"""Round-robin tournament: Random vs Minimax vs Alpha-Beta vs MCTS.

  python run_tournament.py                  # default roster, 40 games/pair
  python run_tournament.py --games 100      # tighter statistics
  python run_tournament.py --roster extended
  python run_tournament.py --quick          # smoke test (~1 min)
"""
import argparse
import json
import os
import time

from stratnet4 import (RandomAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent,
                       round_robin, print_leaderboard)


def build_roster(args):
    s = args.seed
    roster = [
        RandomAgent(seed=s),
        MinimaxAgent(depth=args.mm_depth, seed=s),
        AlphaBetaAgent(depth=args.ab_depth, seed=s),
        MCTSAgent(iterations=args.mcts_iters, seed=s),
    ]
    if args.roster == "extended":
        roster += [
            AlphaBetaAgent(depth=args.ab_depth + 2, seed=s),
            MCTSAgent(iterations=args.mcts_iters * 4, smart_rollout=True, seed=s),
        ]
    return roster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40, help="games per pair (rounded to even)")
    ap.add_argument("--opening-plies", type=int, default=2, help="random opening moves")
    ap.add_argument("--mm-depth", type=int, default=4)
    ap.add_argument("--ab-depth", type=int, default=4)
    ap.add_argument("--mcts-iters", type=int, default=1000)
    ap.add_argument("--roster", choices=["default", "extended"], default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/tournament.json")
    args = ap.parse_args()
    if args.quick:
        args.games, args.mcts_iters = 4, 300

    agents = build_roster(args)
    print("Roster:", ", ".join(a.name for a in agents))
    t0 = time.time()
    res = round_robin(agents, args.games, args.opening_plies, args.seed)
    res["config"] = vars(args)
    print_leaderboard(res)
    print(f"\nFinished in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()

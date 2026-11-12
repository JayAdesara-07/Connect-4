"""Search-efficiency & accuracy benchmarks.

  A. pruning    : Minimax vs Alpha-Beta at equal depth -> nodes, time, pruning ratio,
                  and a check that both return the SAME game-theoretic value.
  B. budget     : round-robin under equal wall-clock budgets per move
                  (iterative-deepening search vs MCTS) -> strength vs compute.
  C. accuracy   : move quality vs a deep Alpha-Beta reference + tactical test
                  (take the win-in-1 / block the loss-in-1).

  python run_benchmarks.py              # full run (~10-20 min)
  python run_benchmarks.py --quick      # smoke test (~1-2 min)
  python run_benchmarks.py --only pruning budget
"""
import argparse
import json
import os
import random
import time

from stratnet4 import (Board, RandomAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent,
                       round_robin, random_position)
from stratnet4.agents import WIN


def sample_positions(n, seed, lo=4, hi=18):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        seq = random_position(rng, rng.randint(lo, hi))
        if seq:
            out.append(Board.from_moves(seq))
    return out


# ---------------------------------------------------------------- A. pruning
def bench_pruning(args):
    print("\n[A] Pruning efficiency (same depth, same heuristic)")
    positions = sample_positions(args.positions, args.seed)
    rows = []
    for d in range(1, args.ab_max_depth + 1):
        row = {"depth": d}
        scores = {}
        for cls, label, dmax in ((MinimaxAgent, "minimax", args.mm_max_depth),
                                 (AlphaBetaAgent, "alphabeta", args.ab_max_depth)):
            if d > dmax:
                continue
            ag = cls(depth=d, seed=0)
            nodes, secs, sc = [], [], []
            for b in positions:
                ag.choose(b)
                nodes.append(ag.last_stats["nodes"])
                secs.append(ag.last_stats["time"])
                sc.append(ag.last_stats["score"])
            row[f"{label}_nodes"] = sum(nodes) / len(nodes)
            row[f"{label}_ms"] = 1000 * sum(secs) / len(secs)
            scores[label] = sc
        if "minimax_nodes" in row:
            row["node_reduction_pct"] = 100 * (1 - row["alphabeta_nodes"] / row["minimax_nodes"])
            row["speedup_x"] = row["minimax_ms"] / row["alphabeta_ms"]
            row["value_match_pct"] = 100 * sum(
                a == b for a, b in zip(scores["minimax"], scores["alphabeta"])) / len(positions)
        rows.append(row)
        print("  depth", d, {k: round(v, 1) for k, v in row.items() if k != "depth"})
    return {"positions": len(positions), "rows": rows}


# ----------------------------------------------------------------- B. budget
def bench_budget(args):
    print("\n[B] Strength under equal per-move time budgets")
    out = []
    for t in args.budgets:
        print(f"  budget = {t}s/move")
        agents = [
            RandomAgent(seed=args.seed),
            MinimaxAgent(depth=14, time_limit=t, seed=args.seed),
            AlphaBetaAgent(depth=14, time_limit=t, seed=args.seed),
            MCTSAgent(iterations=None, time_limit=t, seed=args.seed),
        ]
        res = round_robin(agents, args.budget_games, 2, args.seed)
        for r in res["rows"]:
            print(f"    {r['agent']:<22} score={r['score_pct']:5.1f}%  elo={r['elo']:6.0f}  "
                  f"depth={r['avg_depth']}  nodes/move={r['nodes_per_move']:.0f}")
        out.append({"budget_s": t, **res})
    return out


# --------------------------------------------------------------- C. accuracy
def _tactical_positions(n, seed):
    """Positions where the mover has a win-in-1 ('win') or must block ('block')."""
    rng = random.Random(seed)
    wins, blocks = [], []
    tries = 0
    while (len(wins) < n or len(blocks) < n) and tries < 200000:
        tries += 1
        seq = random_position(rng, rng.randint(6, 22))
        if not seq:
            continue
        b = Board.from_moves(seq)
        lm = b.legal_moves()
        if not lm:
            continue
        w = [c for c in lm if b.is_winning_move(c)]
        o = [c for c in lm if b.opp_would_win(c)]
        if w and len(wins) < n:
            wins.append((b, set(w)))
        elif o and not w and len(o) == 1 and len(blocks) < n:
            blocks.append((b, set(o)))
    return wins, blocks


def bench_accuracy(args):
    print("\n[C] Move accuracy vs deep Alpha-Beta reference + tactical test")
    ref = AlphaBetaAgent(depth=args.ref_depth, seed=0)
    positions = sample_positions(args.acc_positions, args.seed + 1, lo=6, hi=20)
    refs = []
    print(f"  computing reference values (AB depth {args.ref_depth}) for {len(positions)} positions ...")
    for b in positions:
        refs.append(ref.score_moves(b, args.ref_depth))
    wins, blocks = _tactical_positions(args.tactical, args.seed + 2)

    contenders = [
        RandomAgent(seed=args.seed),
        AlphaBetaAgent(depth=1, seed=args.seed),
        AlphaBetaAgent(depth=2, seed=args.seed),
        AlphaBetaAgent(depth=4, seed=args.seed),
        MinimaxAgent(depth=4, seed=args.seed),
    ]
    for n in args.mcts_sweep:
        contenders.append(MCTSAgent(iterations=n, seed=args.seed))
    contenders.append(MCTSAgent(iterations=args.mcts_sweep[-1], smart_rollout=True, seed=args.seed))

    rows = []
    for ag in contenders:
        opt = loss_pen = 0
        ms = []
        for b, sc in zip(positions, refs):
            mv = ag.choose(b)
            ms.append(1000 * ag.last_stats["time"])
            best = max(sc.values())
            opt += sc[mv] == best
            loss_pen += (best - sc[mv]) >= WIN // 2   # outcome class changed (win->not-win / not-loss->loss)
        tact = {}
        for label, group in (("win1", wins), ("block1", blocks)):
            ok = sum(ag.choose(b) in good for b, good in group)
            tact[label] = 100 * ok / max(1, len(group))
        row = {
            "agent": ag.name,
            "ref_optimal_pct": 100 * opt / len(positions),
            "blunder_pct": 100 * loss_pen / len(positions),
            "win_in_1_pct": tact["win1"],
            "block_in_1_pct": tact["block1"],
            "ms_per_move": sum(ms) / len(ms),
        }
        rows.append(row)
        print("  ", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
    return {"ref_depth": args.ref_depth, "positions": len(positions),
            "tactical_n": [len(wins), len(blocks)], "rows": rows}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["pruning", "budget", "accuracy"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--mm-max-depth", type=int, default=5)
    ap.add_argument("--ab-max-depth", type=int, default=7)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.2])
    ap.add_argument("--budget-games", type=int, default=12)
    ap.add_argument("--ref-depth", type=int, default=6)
    ap.add_argument("--acc-positions", type=int, default=30)
    ap.add_argument("--tactical", type=int, default=30)
    ap.add_argument("--mcts-sweep", type=int, nargs="+", default=[100, 500, 2000])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    if args.quick:
        args.positions, args.mm_max_depth, args.ab_max_depth = 8, 4, 5
        args.budgets, args.budget_games = [0.01, 0.05], 4
        args.ref_depth, args.acc_positions, args.tactical = 4, 8, 8
        args.mcts_sweep = [100, 400]

    todo = args.only or ["pruning", "budget", "accuracy"]
    os.makedirs(args.outdir, exist_ok=True)
    fns = {"pruning": bench_pruning, "budget": bench_budget, "accuracy": bench_accuracy}
    for name in todo:
        t0 = time.time()
        res = fns[name](args)
        with open(os.path.join(args.outdir, f"bench_{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> saved results/bench_{name}.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

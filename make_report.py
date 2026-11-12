"""Turn results/*.json into PNG plots + results/REPORT.md.

  python make_report.py
Needs matplotlib for the plots (pip install matplotlib). The Markdown report
is produced even without it.
"""
import json
import os
import sys

R = "results"


def load(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed -> skipping PNG plots (pip install matplotlib)")

    tour, prune = load("tournament.json"), load("bench_pruning.json")
    budget, acc = load("bench_budget.json"), load("bench_accuracy.json")
    md = ["# StratNet4 - Results Report\n"]

    # ---------------- tournament
    if tour:
        rows = sorted(tour["rows"], key=lambda r: -r["elo"])
        md += ["## 1. Round-robin tournament\n",
               md_table(["Agent", "Score %", "W", "D", "L", "Elo", "ms/move", "nodes/move", "avg depth"],
                        [[r["agent"], fmt(r["score_pct"]), r["wins"], r["draws"], r["losses"],
                          fmt(r["elo"], 0), fmt(r["ms_per_move"]), fmt(r["nodes_per_move"], 0),
                          fmt(r["avg_depth"])] for r in rows]), ""]
        if plt:
            names, n = tour["agents"], len(tour["agents"])
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            mat = [[(tour["score"][i][j] / tour["games"][i][j] * 100) if i != j and tour["games"][i][j] else float("nan")
                    for j in range(n)] for i in range(n)]
            im = ax[0].imshow(mat, cmap="RdYlGn", vmin=0, vmax=100)
            ax[0].set_xticks(range(n), names, rotation=35, ha="right")
            ax[0].set_yticks(range(n), names)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ax[0].text(j, i, f"{mat[i][j]:.0f}", ha="center", va="center")
            ax[0].set_title("Score % of row agent vs column agent")
            fig.colorbar(im, ax=ax[0])
            srt = sorted(tour["rows"], key=lambda r: r["elo"])
            ax[1].barh([r["agent"] for r in srt], [r["elo"] for r in srt], color="#4c72b0")
            ax[1].set_xlabel("Elo (Bradley-Terry, mean = 1000)")
            ax[1].set_title("Relative playing strength")
            fig.tight_layout()
            fig.savefig(f"{R}/tournament.png", dpi=140)
            plt.close(fig)
            md.append("![tournament](tournament.png)\n")
            # strength vs compute
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for r in tour["rows"]:
                ax.scatter(max(r["ms_per_move"], 0.01), r["elo"], s=70)
                ax.annotate(r["agent"], (max(r["ms_per_move"], 0.01), r["elo"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("mean ms per move (log)")
            ax.set_ylabel("Elo")
            ax.set_title("Compute cost vs playing strength")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(f"{R}/strength_vs_cost.png", dpi=140)
            plt.close(fig)
            md.append("![cost](strength_vs_cost.png)\n")

    # ---------------- pruning
    if prune:
        rows = prune["rows"]
        md += ["## 2. Pruning efficiency (Minimax vs Alpha-Beta, same depth)\n",
               f"Averaged over {prune['positions']} random mid-game positions.\n",
               md_table(["Depth", "Minimax nodes", "AB nodes", "Node reduction %", "Minimax ms", "AB ms",
                         "Speed-up x", "Same value %"],
                        [[r["depth"], fmt(r.get("minimax_nodes"), 0), fmt(r.get("alphabeta_nodes"), 0),
                          fmt(r.get("node_reduction_pct")), fmt(r.get("minimax_ms"), 2),
                          fmt(r.get("alphabeta_ms"), 2), fmt(r.get("speedup_x")),
                          fmt(r.get("value_match_pct"), 0)] for r in rows]), ""]
        if plt:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for key, lab in (("minimax", "Minimax"), ("alphabeta", "Alpha-Beta")):
                pts = [(r["depth"], r[f"{key}_nodes"], r[f"{key}_ms"]) for r in rows if f"{key}_nodes" in r]
                ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=lab)
                ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=lab)
            for a, t in zip(ax, ("Nodes searched per move", "Time per move (ms)")):
                a.set_yscale("log")
                a.set_xlabel("search depth")
                a.set_title(t)
                a.grid(alpha=.3)
                a.legend()
            fig.tight_layout()
            fig.savefig(f"{R}/pruning.png", dpi=140)
            plt.close(fig)
            md.append("![pruning](pruning.png)\n")

    # ---------------- budget
    if budget:
        md += ["## 3. Strength under equal time budgets (per move)\n"]
        table = []
        for b in budget:
            for r in b["rows"]:
                table.append([f"{b['budget_s']}s", r["agent"], fmt(r["score_pct"]), fmt(r["elo"], 0),
                              fmt(r["avg_depth"]), fmt(r["nodes_per_move"], 0)])
        md += [md_table(["Budget", "Agent", "Score %", "Elo", "Avg depth reached", "Nodes or rollouts/move"], table), ""]
        if plt:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            kinds = ["Random", "Minimax", "AlphaBeta", "MCTS"]
            for k in kinds:
                xs, ys = [], []
                for b in budget:
                    for r in b["rows"]:
                        if r["agent"].startswith(k):
                            xs.append(b["budget_s"])
                            ys.append(r["score_pct"])
                ax.plot(xs, ys, "o-", label=k)
            ax.set_xscale("log")
            ax.set_xlabel("time budget per move (s)")
            ax.set_ylabel("score % in the mini-tournament")
            ax.set_title("Strength vs time budget")
            ax.grid(alpha=.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{R}/budget.png", dpi=140)
            plt.close(fig)
            md.append("![budget](budget.png)\n")

    # ---------------- accuracy
    if acc:
        md += ["## 4. Move accuracy\n",
               f"Reference = Alpha-Beta depth {acc['ref_depth']} (exact value of every move) on "
               f"{acc['positions']} positions; tactical tests: {acc['tactical_n'][0]} win-in-1 and "
               f"{acc['tactical_n'][1]} must-block positions.\n",
               md_table(["Agent", "Ref-optimal %", "Blunder %", "Win-in-1 %", "Block-in-1 %", "ms/move"],
                        [[r["agent"], fmt(r["ref_optimal_pct"]), fmt(r["blunder_pct"]), fmt(r["win_in_1_pct"]),
                          fmt(r["block_in_1_pct"]), fmt(r["ms_per_move"], 2)] for r in acc["rows"]]), ""]
        if plt:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for r in acc["rows"]:
                x = max(r["ms_per_move"], 0.01)
                ax.scatter(x, r["ref_optimal_pct"], s=70)
                ax.annotate(r["agent"], (x, r["ref_optimal_pct"]), textcoords="offset points",
                            xytext=(5, 4), fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("ms per move (log)")
            ax.set_ylabel("% moves optimal vs reference")
            ax.set_title("Accuracy vs speed")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(f"{R}/accuracy.png", dpi=140)
            plt.close(fig)
            md.append("![accuracy](accuracy.png)\n")

    if len(md) == 1:
        sys.exit("No results found - run run_tournament.py / run_benchmarks.py first.")
    with open(f"{R}/REPORT.md", "w") as f:
        f.write("\n".join(md))
    print("Wrote results/REPORT.md" + (" and PNG plots" if plt else ""))


if __name__ == "__main__":
    main()

"""Game runner, round-robin tournament and Elo estimation."""
import math
import random
import sys

from .board import Board, MAX_MOVES


def random_position(rng, plies):
    """Play `plies` random non-terminal moves. Returns the move list (or None)."""
    b, seq = Board(), []
    for _ in range(plies):
        moves = [c for c in b.legal_moves() if not b.is_winning_move(c)]
        if not moves:
            return None
        c = rng.choice(moves)
        b.play(c)
        seq.append(c)
    return seq


def play_game(first, second, opening=()):
    """Play one game. `first` plays the side that moves first from the position
    reached after `opening` (if the opening has odd length, `first` is the side
    to move). Returns (winner, n_moves, per_agent_stats) with winner in
    {0 -> first, 1 -> second, None -> draw}."""
    b = Board.from_moves(opening)
    agents = (first, second)
    acc = [{"time": 0.0, "nodes": 0, "moves": 0, "depth": 0, "depth_n": 0} for _ in agents]
    start = b.moves
    winner = None
    while True:
        idx = (b.moves - start) % 2
        ag = agents[idx]
        col = ag.choose(b)
        if not b.can_play(col):
            raise RuntimeError(f"{ag.name} played illegal column {col}")
        st = ag.last_stats
        a = acc[idx]
        a["time"] += st.get("time", 0.0)
        a["nodes"] += st.get("nodes", 0)
        a["moves"] += 1
        if "depth" in st:
            a["depth"] += st["depth"]
            a["depth_n"] += 1
        if b.is_winning_move(col):
            winner = idx
            break
        b.play(col)
        if b.moves == MAX_MOVES:
            break
    return winner, b.moves, acc


def estimate_elo(score, games, base=1000.0):
    """Maximum-likelihood (Bradley-Terry) Elo from pairwise points.
    A one-game virtual draw between every pair keeps ratings finite when an
    agent never wins."""
    n = len(score)
    R = [0.0] * n
    k = math.log(10) / 400
    for _ in range(300):
        for i in range(n):
            s = e = d = 0.0
            for j in range(n):
                if i == j:
                    continue
                g = games[i][j] + 1
                p = 1 / (1 + 10 ** ((R[j] - R[i]) / 400))
                s += score[i][j] + 0.5
                e += g * p
                d += g * p * (1 - p) * k
            R[i] += max(-100, min(100, (s - e) / d))
    mean = sum(R) / n
    return [base + r - mean for r in R]


def round_robin(agents, games_per_pair=20, opening_plies=2, seed=0, verbose=True):
    """Every pair plays `games_per_pair` games (rounded up to even). Games come
    in colour-swapped pairs that share the same random opening, so luck and
    first-move advantage cancel out."""
    rng = random.Random(seed)
    n = len(agents)
    wins = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]
    eff = [{"time": 0.0, "nodes": 0, "moves": 0, "depth": 0, "depth_n": 0} for _ in agents]
    half = (games_per_pair + 1) // 2
    total = half * 2 * n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            for _ in range(half):
                opening = None
                while opening is None:
                    opening = random_position(rng, opening_plies)
                for a, b in ((i, j), (j, i)):
                    w, _, acc = play_game(agents[a], agents[b], opening)
                    games[i][j] += 1
                    games[j][i] += 1
                    if w is None:
                        draws[i][j] += 1
                        draws[j][i] += 1
                    else:
                        winner, loser = (a, b) if w == 0 else (b, a)
                        wins[winner][loser] += 1
                    for idx, who in enumerate((a, b)):
                        for k, v in acc[idx].items():
                            eff[who][k] += v
                    done += 1
                    if verbose:
                        sys.stdout.write(f"\r  games {done}/{total}")
                        sys.stdout.flush()
    if verbose:
        print()
    score = [[wins[i][j] + 0.5 * draws[i][j] for j in range(n)] for i in range(n)]
    points = [sum(r) for r in score]
    played = [sum(r) for r in games]
    elo = estimate_elo(score, games)
    rows = []
    for i, ag in enumerate(agents):
        e = eff[i]
        rows.append({
            "agent": ag.name,
            "points": points[i],
            "games": played[i],
            "score_pct": 100 * points[i] / played[i] if played[i] else 0,
            "wins": sum(wins[i]),
            "draws": sum(draws[i]) // 1,
            "losses": played[i] - sum(wins[i]) - sum(draws[i]),
            "elo": elo[i],
            "ms_per_move": 1000 * e["time"] / max(1, e["moves"]),
            "nodes_per_move": e["nodes"] / max(1, e["moves"]),
            "avg_depth": e["depth"] / e["depth_n"] if e["depth_n"] else None,
        })
    return {
        "agents": [a.name for a in agents],
        "wins": wins, "draws": draws, "games": games, "score": score,
        "rows": rows,
    }


def print_leaderboard(res):
    rows = sorted(res["rows"], key=lambda r: -r["elo"])
    hdr = f"{'Agent':<22}{'Score%':>8}{'W':>5}{'D':>4}{'L':>5}{'Elo':>7}{'ms/move':>10}{'nodes/move':>12}{'depth':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        d = f"{r['avg_depth']:.1f}" if r["avg_depth"] else "-"
        print(f"{r['agent']:<22}{r['score_pct']:>8.1f}{r['wins']:>5}{r['draws']:>4}"
              f"{r['losses']:>5}{r['elo']:>7.0f}{r['ms_per_move']:>10.1f}"
              f"{r['nodes_per_move']:>12.0f}{d:>7}")

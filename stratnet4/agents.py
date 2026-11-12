"""The four StratNet4 agents.

All agents expose:
    agent.name
    agent.choose(board) -> column
    agent.last_stats    -> {"nodes", "time", "depth", "score"} for the last move
`nodes` is the unit of search work: tree nodes visited for Minimax / Alpha-Beta,
rollouts (simulations) for MCTS.
"""
import math
import random
from time import perf_counter

from .board import (Board, HEIGHT, WIDTH, H1, MAX_MOVES, popcount)

WIN = 100_000
INF = 10 ** 9


# --------------------------------------------------------------------------
# Heuristic evaluation (shared by Minimax and Alpha-Beta so comparisons are fair)
# --------------------------------------------------------------------------
def _build_windows():
    wins = []
    for c in range(WIDTH):
        for r in range(HEIGHT):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + i * dc, r + i * dr) for i in range(4)]
                if all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in cells):
                    m = 0
                    for x, y in cells:
                        m |= 1 << (x * H1 + y)
                    wins.append(m)
    return tuple(wins)


WINDOWS = _build_windows()                     # 69 four-cell windows
_CENTER = ((1 << HEIGHT) - 1) << (3 * H1)
_W = (0, 1, 5, 25, 0)


def evaluate(b: Board) -> int:
    """Static score from the point of view of the player to move."""
    mine, theirs = b.pos, b.pos ^ b.mask
    s = 3 * (popcount(mine & _CENTER) - popcount(theirs & _CENTER))
    for w in WINDOWS:
        a, o = popcount(mine & w), popcount(theirs & w)
        if o == 0 and a:
            s += _W[a]
        elif a == 0 and o:
            s -= _W[o]
    return s


# --------------------------------------------------------------------------
class Agent:
    name = "Agent"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.last_stats = {"nodes": 0, "time": 0.0}

    def choose(self, board: Board) -> int:  # pragma: no cover
        raise NotImplementedError


class RandomAgent(Agent):
    name = "Random"

    def choose(self, board):
        t0 = perf_counter()
        c = self.rng.choice(board.legal_moves())
        self.last_stats = {"nodes": 0, "time": perf_counter() - t0}
        return c


# --------------------------------------------------------------------------
class SearchTimeout(Exception):
    pass


class _SearchAgent(Agent):
    """Negamax search with optional alpha-beta pruning and iterative deepening.

    depth only            -> fixed-depth search
    time_limit (seconds)  -> iterative deepening until time runs out
                             (depth then acts as an upper cap, default 14)
    """
    prune = False
    label = "Search"

    def __init__(self, depth=4, time_limit=None, seed=None, name=None):
        super().__init__(seed)
        self.depth = depth
        self.time_limit = time_limit
        self._deadline = None
        self.nodes = 0
        if name:
            self.name = name
        elif time_limit:
            self.name = f"{self.label}(t={time_limit:g}s)"
        else:
            self.name = f"{self.label}(d={depth})"

    # -- core recursive search (negamax) ---------------------------------
    def _negamax(self, b, depth, alpha, beta, ply):
        self.nodes += 1
        if self._deadline is not None and not (self.nodes & 1023) \
                and perf_counter() > self._deadline:
            raise SearchTimeout
        if b.moves == MAX_MOVES:
            return 0
        moves = b.legal_moves()
        for c in moves:                       # cheap tactical shortcut (both agents)
            if b.is_winning_move(c):
                return WIN - ply
        if depth == 0:
            return evaluate(b)
        best = -INF
        for c in moves:
            child = b.copy()
            child.play(c)
            v = -self._negamax(child, depth - 1, -beta, -alpha, ply + 1)
            if v > best:
                best = v
            if self.prune:
                if v > alpha:
                    alpha = v
                if alpha >= beta:
                    break
        return best

    def _root(self, board, depth):
        best, best_moves = -INF, []
        for c in board.legal_moves():
            if board.is_winning_move(c):
                score = WIN
            else:
                child = board.copy()
                child.play(c)
                # alpha = best-1 keeps ties exact while still pruning
                alpha = best - 1 if (self.prune and best > -INF) else -INF
                score = -self._negamax(child, depth - 1, -INF, -alpha, 1)
            if score > best:
                best, best_moves = score, [c]
            elif score == best:
                best_moves.append(c)
        return best, best_moves

    def score_moves(self, board, depth):
        """Exact negamax value of EVERY legal move (full window). Used as the
        reference oracle in the accuracy benchmark."""
        out = {}
        for c in board.legal_moves():
            if board.is_winning_move(c):
                out[c] = WIN
            else:
                child = board.copy()
                child.play(c)
                out[c] = -self._negamax(child, depth - 1, -INF, INF, 1)
        return out

    def choose(self, board):
        t0 = perf_counter()
        self.nodes = 0
        timed = self.time_limit is not None
        self._deadline = None
        best_moves, best_score, reached = board.legal_moves()[:1], 0, 0
        depths = range(1, (self.depth or 14) + 1) if timed else [self.depth]
        for d in depths:
            # depth 1 is trivial: never abort it, so we always have a legal answer
            self._deadline = (t0 + self.time_limit) if (timed and d > 1) else None
            try:
                score, moves = self._root(board, d)
            except SearchTimeout:
                break
            best_moves, best_score, reached = moves, score, d
            if abs(score) >= WIN - 100:        # forced result found
                break
        self._deadline = None
        self.last_stats = {"nodes": self.nodes, "time": perf_counter() - t0,
                           "depth": reached, "score": best_score}
        return self.rng.choice(best_moves)


class MinimaxAgent(_SearchAgent):
    """Plain depth-limited minimax (negamax form), NO pruning."""
    prune = False
    label = "Minimax"


class AlphaBetaAgent(_SearchAgent):
    """Minimax + alpha-beta pruning + centre-first move ordering."""
    prune = True
    label = "AlphaBeta"


# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("board", "parent", "move", "children", "untried",
                 "wins", "visits", "terminal")

    def __init__(self, board, parent=None, move=None):
        self.board, self.parent, self.move = board, parent, move
        self.children = []
        self.wins = 0.0       # reward for the player who MOVED INTO this node
        self.visits = 0
        if move is not None and board.last_mover_won():
            self.terminal = 1.0
        elif board.moves == MAX_MOVES:
            self.terminal = 0.5
        else:
            self.terminal = None
        self.untried = board.legal_moves() if self.terminal is None else []


class MCTSAgent(Agent):
    """UCT Monte-Carlo Tree Search with random (or 'smart') rollouts.

    iterations : fixed simulation budget per move, and/or
    time_limit : wall-clock budget per move (seconds)
    smart_rollout : in playouts, always take an immediate win / block an
                    immediate loss. Slower per rollout, much better signal.
    """

    def __init__(self, iterations=1000, time_limit=None, c=1.41,
                 smart_rollout=False, seed=None, name=None):
        super().__init__(seed)
        self.iterations, self.time_limit = iterations, time_limit
        self.c, self.smart = c, smart_rollout
        tag = "+smart" if smart_rollout else ""
        if name:
            self.name = name
        elif time_limit:
            self.name = f"MCTS{tag}(t={time_limit:g}s)"
        else:
            self.name = f"MCTS{tag}(n={iterations})"

    def _rollout(self, b):
        """Random playout. Returns +1 if the player to move at `b` wins,
        -1 if loses, 0 draw."""
        b = b.copy()
        sign, rng = 1, self.rng
        while True:
            if b.moves == MAX_MOVES:
                return 0
            moves = b.legal_moves()
            if self.smart:
                forced = None
                for m in moves:
                    if b.is_winning_move(m):
                        return sign
                    if forced is None and b.opp_would_win(m):
                        forced = m
                c = forced if forced is not None else rng.choice(moves)
            else:
                c = rng.choice(moves)
                if b.is_winning_move(c):
                    return sign
            b.play(c)
            sign = -sign

    def _select(self, node):
        logn = math.log(node.visits)
        c = self.c
        return max(node.children,
                   key=lambda ch: ch.wins / ch.visits + c * math.sqrt(logn / ch.visits))

    def choose(self, board):
        t0 = perf_counter()
        deadline = t0 + self.time_limit if self.time_limit else None
        limit = self.iterations if self.iterations else float("inf")
        root = _Node(board.copy())
        it = 0
        while it < limit and (deadline is None or perf_counter() < deadline):
            node = root
            while node.terminal is None and not node.untried:      # selection
                node = self._select(node)
            if node.terminal is None:                              # expansion
                m = node.untried.pop(self.rng.randrange(len(node.untried)))
                nb = node.board.copy()
                nb.play(m)
                child = _Node(nb, node, m)
                node.children.append(child)
                node = child
            if node.terminal is not None:                          # simulation
                reward = node.terminal
            else:
                reward = (1 - self._rollout(node.board)) / 2
            while node is not None:                                # backprop
                node.visits += 1
                node.wins += reward
                reward = 1 - reward
                node = node.parent
            it += 1
        top = max(ch.visits for ch in root.children)
        best = self.rng.choice([ch for ch in root.children if ch.visits == top])
        self.last_stats = {"nodes": it, "time": perf_counter() - t0,
                           "score": best.wins / best.visits}
        return best.move

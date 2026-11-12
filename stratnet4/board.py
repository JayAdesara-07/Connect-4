"""Fast bitboard Connect 4 (7 columns x 6 rows).

Each column uses 7 bits (6 cells + 1 sentinel bit) so shifts never wrap
between columns. Bit index of cell = col * 7 + row   (row 0 = bottom).

`pos`  : stones of the player TO MOVE
`mask` : stones of both players
The opponent's stones are therefore `pos ^ mask`.
"""

WIDTH, HEIGHT = 7, 6
H1 = HEIGHT + 1
MAX_MOVES = WIDTH * HEIGHT
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)  # centre-first ordering helps pruning

_BOTTOM = tuple(1 << (c * H1) for c in range(WIDTH))
_TOP = tuple(1 << (HEIGHT - 1 + c * H1) for c in range(WIDTH))
_COL = tuple(((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH))

try:  # Python 3.10+
    popcount = int.bit_count
except AttributeError:  # pragma: no cover
    def popcount(x):
        return bin(x).count("1")


def has_four(p: int) -> bool:
    """True if bitboard p contains four in a row (any direction)."""
    m = p & (p >> H1)            # horizontal
    if m & (m >> (2 * H1)):
        return True
    m = p & (p >> HEIGHT)        # diagonal \
    if m & (m >> (2 * HEIGHT)):
        return True
    m = p & (p >> (HEIGHT + 2))  # diagonal /
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    m = p & (p >> 1)             # vertical
    return bool(m & (m >> 2))


class Board:
    __slots__ = ("pos", "mask", "moves")

    def __init__(self, pos: int = 0, mask: int = 0, moves: int = 0):
        self.pos, self.mask, self.moves = pos, mask, moves

    # ---- construction -------------------------------------------------
    def copy(self) -> "Board":
        return Board(self.pos, self.mask, self.moves)

    @classmethod
    def from_moves(cls, seq):
        b = cls()
        for c in seq:
            if not b.can_play(c):
                raise ValueError(f"illegal move {c}")
            b.play(c)
        return b

    # ---- queries ------------------------------------------------------
    def can_play(self, col: int) -> bool:
        return not (self.mask & _TOP[col])

    def legal_moves(self):
        m = self.mask
        return [c for c in CENTER_ORDER if not m & _TOP[c]]

    def is_full(self) -> bool:
        return self.moves == MAX_MOVES

    def is_winning_move(self, col: int) -> bool:
        """Would the player to move win by playing `col`?"""
        p = self.pos | ((self.mask + _BOTTOM[col]) & _COL[col])
        return has_four(p)

    def opp_would_win(self, col: int) -> bool:
        """Would the opponent win if they played `col` (i.e. we must block)?"""
        p = (self.pos ^ self.mask) | ((self.mask + _BOTTOM[col]) & _COL[col])
        return has_four(p)

    def last_mover_won(self) -> bool:
        return has_four(self.pos ^ self.mask)

    def key(self) -> int:
        return self.pos + self.mask

    def to_move(self) -> int:
        return 1 + (self.moves & 1)

    # ---- mutation -----------------------------------------------------
    def play(self, col: int) -> None:
        self.pos ^= self.mask
        self.mask |= self.mask + _BOTTOM[col]
        self.moves += 1

    # ---- display ------------------------------------------------------
    def render(self) -> str:
        p1 = self.pos if self.moves % 2 == 0 else self.pos ^ self.mask
        p2 = self.mask ^ p1
        rows = []
        for r in range(HEIGHT - 1, -1, -1):
            line = []
            for c in range(WIDTH):
                bit = 1 << (c * H1 + r)
                line.append("X" if p1 & bit else "O" if p2 & bit else ".")
            rows.append(" ".join(line))
        rows.append("0 1 2 3 4 5 6")
        return "\n".join(rows)

    def __repr__(self):
        return self.render()

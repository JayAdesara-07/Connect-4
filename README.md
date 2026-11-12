# StratNet4 — Connect 4 Strategy Arena

Compares four Connect 4 agents — **Random, Minimax, Alpha-Beta, MCTS** — through
round-robin tournaments, search-efficiency measurements and time-budget experiments.

## Project layout
```
stratnet4/
├── stratnet4/
│   ├── board.py     # bitboard engine (fast win detection, 7x6)
│   ├── agents.py    # Random, Minimax, AlphaBeta (+iterative deepening), MCTS (UCT)
│   └── arena.py     # game runner, round-robin, Bradley-Terry Elo
├── run_tournament.py   # round-robin strength comparison
├── run_benchmarks.py   # pruning efficiency / time budgets / accuracy
├── make_report.py      # plots + results/REPORT.md
├── play.py             # play against any agent in the terminal
├── tests/test_stratnet4.py
├── Makefile, requirements.txt
└── results/            # generated output (JSON, PNG, REPORT.md)
```

## Setup (Python 3.9+, 3.10+ recommended)
```bash
cd stratnet4
pip install -r requirements.txt      # only matplotlib, needed just for plots
```

## What to run (in this order)
```bash
python -m unittest discover -s tests -v          # 1. sanity tests (≈1 s)
python run_tournament.py --quick                 # 2. smoke test (≈5 s)

python run_tournament.py --games 40 --roster extended   # 3. main tournament (~5-10 min)
python run_benchmarks.py                                # 4. all benchmarks   (~10-20 min)
python make_report.py                                   # 5. plots + REPORT.md
python play.py --opponent alphabeta --depth 6           # 6. (optional) play it yourself
```
Or simply `make all` (or `make quick` for a fast end-to-end pass).
Everything is seeded (`--seed`), so results are reproducible.
Use `--games 100` for tighter confidence; `run_benchmarks.py --only pruning` to run one experiment.

## Method (what each resume bullet maps to)
| Bullet | Implementation |
|---|---|
| Four AI agents | `agents.py`: Random; Minimax (negamax, no pruning); Alpha-Beta (pruning + centre-first ordering); MCTS (UCT, random or "smart" rollouts). Minimax and Alpha-Beta share one heuristic (69 four-cell windows + centre bonus), so differences come only from the search. |
| Round-robin tournaments | `arena.round_robin`: every pair plays colour-swapped game pairs from the same random 2-ply opening (cancels luck and first-move advantage). Outputs win/draw/loss, score %, Bradley-Terry Elo. |
| Search efficiency per strategy | Every move logs nodes searched (rollouts for MCTS), wall time and depth reached → `ms/move`, `nodes/move`, `strength_vs_cost.png`. |
| Pruning & rollout trade-offs under time budgets | Benchmark A: nodes/time vs depth, with a check that Alpha-Beta returns the *same value* as Minimax. Benchmark B: iterative-deepening Minimax/Alpha-Beta vs time-limited MCTS at 0.01/0.05/0.2 s per move. Benchmark C: % of moves optimal against a depth-6 reference, blunder rate, win-in-1/block-in-1 tests, MCTS rollout sweep (100/500/2000) and smart vs random rollouts. |
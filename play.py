"""Play Connect 4 against any StratNet4 agent in the terminal.

  python play.py --opponent alphabeta --depth 6
  python play.py --opponent mcts --iters 3000 --second     # AI moves first
Opponents: random | minimax | alphabeta | mcts
"""
import argparse

from stratnet4 import Board, RandomAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="alphabeta", choices=["random", "minimax", "alphabeta", "mcts"])
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--second", action="store_true", help="let the AI move first")
    a = ap.parse_args()
    ai = {"random": lambda: RandomAgent(),
          "minimax": lambda: MinimaxAgent(depth=a.depth),
          "alphabeta": lambda: AlphaBetaAgent(depth=a.depth),
          "mcts": lambda: MCTSAgent(iterations=a.iters)}[a.opponent]()
    human_turn = not a.second
    b = Board()
    print(f"You are {'X (first)' if human_turn else 'O (second)'} vs {ai.name}\n")
    while True:
        print(b.render(), "\n")
        if human_turn:
            try:
                c = int(input("Your column (0-6): "))
            except (ValueError, EOFError):
                print("Enter a number 0-6.")
                continue
            if not (0 <= c <= 6) or not b.can_play(c):
                print("Illegal move.")
                continue
            who = "You"
        else:
            c = ai.choose(b)
            s = ai.last_stats
            print(f"{ai.name} plays {c}  ({s['nodes']} nodes, {1000 * s['time']:.0f} ms)")
            who = ai.name
        if b.is_winning_move(c):
            b.play(c)
            print(b.render(), f"\n{who} wins!")
            return
        b.play(c)
        if b.is_full():
            print(b.render(), "\nDraw.")
            return
        human_turn = not human_turn


if __name__ == "__main__":
    main()

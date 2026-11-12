PY ?= python3

test:        ; $(PY) -m unittest discover -s tests -v
quick:       ; $(PY) run_tournament.py --quick && $(PY) run_benchmarks.py --quick && $(PY) make_report.py
tournament:  ; $(PY) run_tournament.py --games 40 --roster extended
bench:       ; $(PY) run_benchmarks.py
report:      ; $(PY) make_report.py
all: test tournament bench report

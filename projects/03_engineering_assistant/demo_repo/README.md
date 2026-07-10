# Demo Repo — Inventory Manager

A small, real, multi-file fixture repo for Project 3's engineering assistant to operate on
(ARCHITECTURE.md "The demo repo"). Deliberately richer than Module 17's `mini_calculator` (a
2-file fixture explicitly named in that module's own report as too small to exercise multi-file
patches or "unrelated file changes" rejection).

- `inventory/stock.py` — stock tracking. Contains one real, currently-failing test: `remove_stock()`
  doesn't validate against removing more than the current quantity.
- `inventory/pricing.py` — pricing calculations, no known bugs.
- `inventory/reports.py` — summary reporting, imports from both `stock.py` and `pricing.py`
  (a real cross-file dependency).

Run `python -m pytest tests -q` from this directory to see the real failure.

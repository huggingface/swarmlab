#!/usr/bin/env python
"""M2 phase-1 Flag Game report. Read-only over the runs dir. Usage: m2_report.py RUNS_DIR [--out FILE]

Kept for the M2 notes; the code now lives in `swarmlab/report.py` (`swarmlab report RUNS_DIR`).
"""
from __future__ import annotations

import argparse

from swarmlab.report import write_report

TITLE = "M2 phase-1 Flag Game report"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    print(write_report(a.runs_dir, a.out, TITLE))


if __name__ == "__main__":
    main()

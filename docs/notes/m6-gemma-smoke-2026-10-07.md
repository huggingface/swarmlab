# Gemma 4 26B-A4B smoke arms (2026-10-07, via HF router → DeepInfra)

| arm | calls | errors | tool calls ok | probes ok | accuracy r1-3 | posts | spend |
|---|---|---|---|---|---|---|---|
| gemma-image (N=4, image mode, broadcast) | 36 | 0 | 24/24 | 12/12 | 0.00, 0.25, 0.25 | 0 | $0.014 |
| gemma-manager (N=4 incl. blind a000, star) | 40 | 0 | 28/28 | 12/12 | 0.00, 0.33, 0.00 | 0 member posts; manager posted nothing | $0.016 |
| gemma-image (earlier Python run) | 37 | 0 | 25/25 | — | 0.00, 0.50, 0.00 | 1 | $0.015 |

Images and native tool calls work through the router; latency median ~14 s. Gemma 4 26B-A4B does not communicate unprompted in three rounds, so the manager protocol had nothing to relay, and its accuracy on 12 px synthetic candidate images is at or below chance. The paper's forced per-turn JSON report (M6 `report_json`) is the mechanism that makes members report; the faithful run will show whether recognition of real flags from a crop is better than matching synthetic grids. Run dirs under `/data/workspaces/ultra-harness/smoke/20261007-165001/` and `smoke/gemma/`.

# Flag Game image modality: first real run (2026-10-07)

`tools/real_smoke.py --arms haiku-image`: Haiku 4.5, N=4, 3 rounds, broadcast, `modality: image`, `max_calls: 5`, belief probe every round. Spend $0.092 ($0.070 swarm + $0.022 measurement) against a $0.43 worst case.

| calls | errors | tool calls ok | yields | accuracy by round | probe parse | posts |
|---|---|---|---|---|---|---|
| 33 | 0 | 28/28 | 12 end_turn | 0.75, 0.75, 1.00 | 12/12 | 0 |

Images (one PNG per candidate plus the crop, 12 px cells) were accepted by the Anthropic adapter and the model acted on them: the agents' guesses started at the same 0.75 the text variant gives from crops alone. Nobody posted, so the "do posts mention visual features" check had no data; convergence came through `collective_status`, as in text mode. Run dir: `/data/workspaces/ultra-harness/smoke/20261007-150215/`.

Next: `experiments/m5_flag_image.yaml` (image vs text, N=8, 8 rounds, 3 seeds, Haiku) for a paired modality comparison; worst case $19, realistic about $5.

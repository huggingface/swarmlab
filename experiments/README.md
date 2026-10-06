# Experiments

- `m2_flaggame.yaml` — M2 Flag Game proving experiment, phase 1 (N=16, 10 rounds, 3 seeds): broadcast vs gossip(k=1), phase-commit vs sequential (`-seq` arms), Haiku 4.5 vs Qwen3.5-9B. Probes every round. Phase 2 (N=64) and the paired-run influence measurement follow once phase 1 has validated the pipeline.

Cost model (from the 2026-10-06 smoke, full-history memory): a Haiku agent-round costs about $0.009 in round 1 and grows roughly $0.004 per round as context accumulates; broadcast at N=16 adds about 16 posts per read. Expected per run: Haiku broadcast about $8 (probes are a third of it, since a probe re-sends the agent's full context), Haiku gossip about $5, Qwen about $0.6. The framework's `swarmlab estimate` is a flat worst case (20 calls per turn × 3,000 prompt tokens) and does not model growth; treat it as a ceiling check, not a forecast.

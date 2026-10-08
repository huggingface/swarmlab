# Cascade on Qwen3.8-27B (2026-10-08)

Spec: `experiments/cascade.yaml` (guide: [docs/guide/cascade.md](../guide/cascade.md)). The model is `hf:Qwen/Qwen3.8-27B:deepinfra` with thinking off and provider-default temperature. Every arm has 20 workers, truth `output_only`, 70% signals, and the first four signals wrong in `*_stress` arms.

## Main result (seeds 1–40)

| Arm | Wrong, positions 5–20 | Wrong at position 20 | Against own signal, positions 5–20 |
|---|---|---|---|
| `board_stress` | 77% | 57% (23/40 runs) | 52% |
| `noboard_stress` | 38% | 28% (11/40 runs) | 7% |
| signal points to the wrong answer | 31% | – | – |

The board effect replicates but is weaker than in the source. The source's Haiku 4.5 board line stays near 93% wrong through position 30. Qwen's is 93–100% wrong at positions 5–6, then falls to about 57% by position 20. Without the board, workers commit a few points more often to `reads_transcript` than their signals alone would explain: they follow their own signal 93% of the time, and every deviation goes that way.

## Single-seed arms (seed 3)

| Arm | Right, positions 5–20 | Right, positions 12–20 |
|---|---|---|
| `noboard_stress` | 10/16 | 6/9 |
| `board_stress` | 2/16 | 2/9 |
| `board_normal` | 8/16 | 6/9 |
| `board_quote_stress` | 4/16 | 4/9 |
| ideal observer (all signals so far) | – | 8/9 |

The pooled signals favour the truth from position 12 on. Behaviour seen in the seed-3 posts:
- **Misreported results.** On free-form boards, some workers posted a probe result different from the one they received, always ACCEPTED turned into REJECTED. `board_stress` W04 counted its own ACCEPTED as a fourth REJECTED, and W19 posted "Result: REJECTED". `board_normal` W06 and W10 did the same. The board's running tally drifted with it.
- **The E-INTEGRITY argument.** From W06–W07 on, posts argue that a rejection carrying a specific integrity code outweighs an acceptance, and later posts repeat it.
- **The quote rule.** Every post quoted the true result, but commitments still leaned `reads_transcript`.

## Operational notes

- **Cerebras outage.** `hf:Qwen/Qwen3.8-27B:cerebras` was paused by the HF router's circuit breaker (HTTP 503) for part of the day, and 3–5 workers per run ended without an answer. Those runs were discarded and the experiment moved to DeepInfra.
- **Truncation.** With `max_tokens: 1024`, 34 of 800 `board_stress` turns (40 seeds) were truncated once and answered on the retry. 10 more were truncated twice and have no commitment. `noboard_stress` had none.
- **Spend.** The four seed-3 arms plus the 40-seed pair cost $2.55, including the discarded Cerebras runs.

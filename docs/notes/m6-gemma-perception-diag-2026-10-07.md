# Gemma 4 26B-A4B: perception vs reasoning on synthetic Flag Game crops (2026-10-07)

Round-1, crop-only accuracy (no communication yet), 4 agents, seeds 1-3, broadcast, via HF router → DeepInfra. Total spend $0.018.

| condition | accuracy by seed | mean |
|---|---|---|
| perfect-perception baseline (scripted `Silent`: random among candidates containing the crop) | .50 .75 .75 | 0.67 |
| text grids | .75 .25 .50 | 0.50 |
| image, 12 px cells (144×96 px candidates) | .00 .50 .25 | 0.25 |
| image, 32 px cells (384×256 px) | .25 .25 .00 | 0.17 |
| image + text grids (`image_text_hint`) | .75 .25 .50 | 0.50 |

Reading: on text, Gemma loses some reasoning against the perfect-perception ceiling (0.50 vs 0.67). On images alone it is near chance (8 candidates → 0.125) and resolution does not help, so visual matching of abstract colour grids is the failure, not image size. With text available it ignores the images (0.50 = text score). Haiku 4.5 scores 0.75 in both modalities on the same seeds. The paper does not vary world knowledge (28 real flags in every condition); its "informative crop" criterion (true country in 10/10 isolated probes) is this same crop-only measurement. The synthetic mode is therefore a knowledge-free contrast the paper lacks; the real-flag faithful mode (M6) tests the knowledge-driven task. Follow-up: a crop-informativeness pass (isolated probes per agent before the swarm runs) as a built-in pre-run step. Runs under `$AM_LOCAL/runs/diag/`.

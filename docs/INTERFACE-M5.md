# swarmlab M5 interface contract: Flag Game image modality

Scope: DESIGN.md component 4, "modality: text-rendered crops by default, image crops as a parameter". Builds on INTERFACE.md §7-§8 (Part, Observation, FlagGame) and INTERFACE-M1b §1 (image parts in providers). Nothing here changes the spec hash of a text-mode run.

## 1. World parameters

`FlagGame(modality: Literal["text", "image"] = "text", cell_px: int = 12, image_text_hint: bool = False)`.

- `text` is today's behaviour, byte for byte.
- `image`: `observe()` returns, in order: one text part naming the candidates and describing the task framing ("Candidates A..H are shown as images in that order; your crop follows."), one image part per candidate (rendered flag), one text part "Your crop:" and one image part for the crop. Candidate images carry no label inside the image; the order in the message is the labelling (state this in the text part). With `image_text_hint=True` the text grids are included as well (an ablation condition, and what the fake provider and scripted agents read).
- Private information (`crop_y`, `crop_x`) and everything else are unchanged. `description()` says the candidates and crop are images.

## 2. Rendering

`swarmlab/world/render.py`: `grid_to_png(rows: list[str], palette: dict[str, tuple[int,int,int]], cell_px: int) -> bytes`, a dependency-free PNG encoder (zlib + struct, RGB, no interlace), plus `png_size(bytes) -> (w, h)` and a `decode_png_rgb(bytes) -> list[list[tuple]]` used by tests (solid-colour cells, so decoding is exact). Colour letters map to a fixed palette shared with the viewer (`r,g,b,y,k,w,o,p,c,m,n,t`). The crop image uses the same cell size as the candidates. Rendering is deterministic.

## 3. Providers

Image parts already exist in `ChatMessage.content` and the adapters (Anthropic image blocks, OpenAI-compatible `image_url` data URLs). Verify both paths with stubbed transports in tests (correct base64, media type `image/png`, ordering preserved). `Provider.estimate_prompt_tokens` counts an image part as `max(100, w*h/750)` tokens (Anthropic's formula; conservative for others). A text-only model receiving images gets a provider 400; `swarmlab preflight` must surface this as "model rejects image input" and the README must say that image mode needs a vision model (Haiku 4.5 yes; Qwen3.5-9B with `--language-model-only` no; Qwen VL variants on the router or vLLM yes).

## 4. Participants, probes, exports

- `LLMAgent` passes image parts through unchanged (already the contract). Under `memory: window` images are dropped with their rounds like text.
- `fake:reader` and scripted participants require `image_text_hint=True` in image mode; without it they raise a clear error at bind.
- Probes are unchanged (text question; the probe context carries the image parts).
- Export sessions represent an image part as `[image: PNG w×h]` text; the Parquet `inference` table keeps the request hash only. Viewer: unchanged (it draws grids from the world snapshot).

## 5. Real check and acceptance

1. Text mode unchanged: spec hash, observation text, and the M2 report reproduction all identical.
2. Image mode observation: parts in the documented order; PNG decodes back to the grid exactly; the crop image is a sub-image of the truth's image at the crop position; `image_text_hint` adds the text grids after each image.
3. Both adapters serialise image parts correctly (stubbed).
4. `fake:reader` with `image_text_hint=True` plays a full run deterministically; replay and fork work.
5. `tools/real_smoke.py` gains an `haiku-image` arm (N=4, 3 rounds, image mode, hard ceiling $0.50) that reports per-turn cost, whether the model's posts mention visual features, and accuracy; not run by the implementer.
6. Documentation: README "Flag Game modalities" paragraph; `experiments/` gets `m5_flag_image.yaml` with Haiku image vs text arms at N=8 for a later approval.

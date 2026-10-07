# Field notes 7-11: report, export, publish, viewer

Branch `field-notes-report` (not merged). Fixes for items 7-11 of
`docs/notes/field-notes-crowding-pilot-2026-10-07.md`, one commit each.

| item | change | tests |
|---|---|---|
| 7 | `report`: "Protocol health" section per arm for any world (turn end kinds, errored turns with the first error, finish reasons, retries, latency median/p90/max, calls per turn, cost per round, rejected tool calls, refused world actions, skipped probes); trajectories list every logged metric; Flag Game sections only for FlagGame worlds (or committed guesses), a "Coloring (final grid)" section only for ColoringGrid | `test_report.py::test_protocol_health_and_metrics_for_a_custom_world`, `::test_first_error_line_and_percentile`; the M2 reproduction drops the added sections and rows |
| 8 | `export_run(raw=False)` / `export --no-raw` / `publish --no-raw`; raw copy in one pass that reports every unreadable file and raises `ExportError` without writing `run.json`; unreadable blobs read for tables/sessions listed in `run.json["unreadable_blobs"]`; `publish` raises `PublishFailed` naming the step (export, repo, upload), CLI exit 1 | `test_export.py::test_raw_copy_io_error_fails_the_export`, `::test_no_raw_export_reports_unreadable_blobs`, `test_publish.py::test_publish_failures_exit_non_zero_and_name_the_step`, `::test_publish_no_raw_uploads_tables_and_sessions_only` |
| 9 | sessions: an unreadable request blob no longer resets the overlap reference (root cause of the duplicated `Round 1.`: one request blob of crowding-pilot baseline s3 a008, round 2, gives `[Errno 5]` on the bucket mount, so the round-3 request was emitted in full); its response is placed where the next request puts it. `tests/pi_session.py` checks round markers strictly increase | `test_export.py::test_unreadable_request_blob_does_not_duplicate_rounds`, `::test_validator_rejects_a_repeated_round_marker` |
| 10 | README "Analysing results" / "Analysis patterns": per-round metrics, `Outcome.feedback` -> `actions` table, sessions for reasoning text, a `needs_truth`/`set_truth` metric sample | `test_report.py::test_readme_truth_metric_example_runs` (runs the README sample) |
| 11 | `World.render_state() -> dict | None` (default None); `viewer/build.py` restores a fresh world from each round's snapshot into `world_states`; the page's "World state" panel draws `grid` (+ other 2-D lists with `palette`), tables and key/values; ColoringGrid returns current and target grids | `test_viewer.py::test_world_state_panel_for_coloring_and_a_custom_world`; screenshots with `tools/viewer-check` (light and dark) checked by eye |

`cli.py`: only the `export` and `publish` commands (a `--no-raw` option each, passing `raw=`,
and a warning line). Test fixtures for a custom world (`SiteWorld`, `site_script`,
`CrowdMetric`, `Crasher`, `site_experiment`) are in `tests/helpers.py`.

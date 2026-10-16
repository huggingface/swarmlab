"""HF Jobs placement with a co-located vLLM server, or against API models (DESIGN.md §12;
docs/handoff/WP8.md).

`launch.py` plans, stages and submits a job (`swarmlab job run`; vLLM mode or API mode);
`profiles.py` holds the per-model vLLM serve flags (explicit `--vllm-flags` > profile > Qwen
default); `bootstrap.sh` runs inside the job (vLLM or not + `swarmlab run` per arm x seed + bucket
sync + manifest, or the grade); `remote.py` reads jobs and fetches run dirs back (`swarmlab job
status|logs|fetch`); `programbench.py` plans and launches the grade job of a ProgramBench run and
writes its score back (`swarmlab programbench grade|fetch-grade`); `grade.py` runs inside that
job (the official evaluator on a local container class).
"""
from .launch import JobPlan, describe, hf_command, plan_job, stage, submit
from .remote import fetch, find_run, logs, status

__all__ = ["JobPlan", "describe", "fetch", "find_run", "hf_command", "logs",
           "plan_job", "stage", "status", "submit"]

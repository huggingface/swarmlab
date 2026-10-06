"""HF Jobs placement with a co-located vLLM server (DESIGN.md §12; docs/handoff/WP8.md).

`launch.py` plans, stages and submits a job (`swarmlab job run`); `bootstrap.sh` runs inside it
(vLLM + `swarmlab run` per arm x seed + bucket sync + manifest); `remote.py` reads jobs and
fetches run dirs back (`swarmlab job status|logs|fetch`).
"""
from .launch import JobPlan, describe, hf_command, plan_job, stage, submit
from .remote import fetch, find_run, logs, status

__all__ = ["JobPlan", "describe", "fetch", "find_run", "hf_command", "logs",
           "plan_job", "stage", "status", "submit"]

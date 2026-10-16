# ProgramBench

The `programbench` world runs one [ProgramBench](https://arxiv.org/abs/2605.03546) task: the agent gets a workspace holding a compiled reference program (`./executable`, execute-only), its bundled documentation and test assets, and no source code and no internet. It has to write an independent implementation plus a `compile.sh` that builds a new `./executable`, and the whole workspace is the deliverable. Module: `swarmlab/world/programbench.py` (its docstring is the exact reference). Back to the [docs index](../README.md).

The world reproduces the official harness's environment (mini-swe-agent 2.4.6 with its `programbench.yaml` config), so a solo run is a leaderboard-style submission: the task image, the workspace, the command semantics, the submission rule and the limits are the benchmark's, and only the agent harness differs. Grading is a separate step: the world keeps the workspace in every round's snapshot and never runs the hidden tests.

## Install

```
pip install "swarmlab[programbench]"
```

The extra pins `programbench==1.2.5` (task metadata now; the grader in a later step). Building the world, replaying a run, the viewer and reports work without it; only the run-start task check (`check_task`) imports `programbench`. mini-swe-agent, the faithful participant, comes with its own extra (see [Faithful participant](#faithful-participant)); the world's task text is copied from its 2.4.6 release.

## Parameters

`world: {type: programbench, params: {...}}`

| param | default | meaning |
|---|---|---|
| `task` | required | instance id, e.g. `abishekvashok__cmatrix.5c082c6`; the image is `programbench/<task with __ -> _1776_>:task_cleanroom_v6` |
| `backend` | `local` | `local`: commands run on this machine (swarmlab inside the task image, the HF Jobs shape); `docker`: the world starts the task image and runs each command with `docker exec` |
| `workspace` | `/workspace` | the task directory, every command's working directory |
| `user` | `agent` | who runs the commands: `--user` under Docker; locally, when swarmlab runs as root and the user exists, commands drop to it with `setpriv` (else they run as the current user) |
| `step_limit` | 1000 | the benchmark's model-call limit; enforced by the participant, and by the run-start check below |
| `wall_limit_s` | 21600 | the 6 h wall-clock limit; past it the run is terminal and `bash` is refused |
| `command_timeout_s` | 180 | per-command timeout |
| `lockdown` | true | job-mode network and sudo lockdown (local backend, as root only; below) |
| `pin_hosts` | `[api.anthropic.com, huggingface.co, hf.co]` | the hosts that stay reachable under lockdown |
| `max_output_bytes` | 1000000 | safety cap on one command's output (head and tail kept around one elision line); `null` disables it |
| `cpus`, `memory_gb` | null | the resources the task text states; docker defaults to the official 20 CPUs and 60 GB, the local backend detects them from the cgroup under lockdown |
| `snapshot_exclude` | `[]` | fnmatch patterns left out of the workspace snapshot (e.g. `target`); by default the whole workspace is kept, since it is the submission |
| `check_task` | true | at run start, refuse a task the `programbench` package does not know |

Every param is in the spec and its hash. That includes `backend`: a run inside the task image and a Docker run are different experiments.

## The two backends

- **`local`** runs `bash -lc <command>` directly, in its own session, with the output going to a temp file. This is the shape for an HF Job that runs swarmlab inside the task image (Jobs cannot run Docker). The command environment is the inherited one minus anything secret-looking (`ANTHROPIC_*`, `HF_TOKEN`, `*_KEY`, `*_TOKEN`, `*_SECRET`, ...) and minus swarmlab's own virtualenv, plus the baseline variables and, with a known CPU count, `MAKEFLAGS=-j<n>`, `CARGO_BUILD_JOBS`, `OMP_NUM_THREADS` and `GOMAXPROCS`.
- **`docker`** runs `docker run -d --network none --cpus 20 --memory 60g --memory-swap 60g --user agent --cap-drop SYS_PTRACE <image> sleep 7h` at the first command, then `docker exec -w /workspace -e PAGER=cat ... bash -lc <command>` per call, as mini-swe-agent does. The container is removed at the end of the last round (terminal, or the run's `max_rounds`); a run that stops another way (budget, stop condition, crash) relies on a cleanup hook that runs at interpreter exit.

Neither backend starts before the first command: `swarmlab prompts`, `replay`, `report` and the viewer never start a container or lock a machine down.

Under lockdown (local backend, `lockdown: true`, swarmlab running as root), the first command first pins each `pin_hosts` host in `/etc/hosts`, points `/etc/resolv.conf` at `0.0.0.0`, and removes the agent user's sudo rules and the sudo binary's permissions. Nothing happens when swarmlab is not root.

## Official semantics kept

- One tool, `bash(command)`, with mini-swe-agent's schema. Its feedback is the baseline's result dict, `{"output", "returncode", "exception_info"}`: stdout and stderr merged, never truncated for the agent (the 5000/5000-character head and tail rendering past 10000 characters is the participant's job). The only cap is `max_output_bytes`.
- Every call is a fresh `bash -lc` in the workspace, so `cd` and `export` do not persist. The environment carries `PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1`.
- On a timeout the agent sees the output so far, return code -1 and the baseline's `An error occurred while executing the command: ...` text. The process keeps running, as under Docker, where only the waiting client is killed.
- A command whose first non-blank output line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, with return code 0, submits. The run is then terminal and any later `bash` is refused (`submitted`). There is no `submit` tool. At a limit, the workspace is submitted as it is.
- Before the first command the world sets the baseline's git identity in the workspace repo (not a step).
- The task text (`World.description()`, what an `llm` participant sees) is the baseline's system template followed by its instance template, with the same machine details filled in. One sentence is added at the end (below).

## Documented deviations

- Network: DNS blackhole + pinned API/Hub hosts + sudo removed, instead of `--network none`. Raw-IP connections remain possible. Unauthorized access is found from the event log, not a report.
- Resources: default `cpu-upgrade` (8 vCPU, 32 GB) vs official 20 CPU / 60 GB; flavor is a launch option. One added prompt sentence states the real CPU count and memory; MAKEFLAGS, CARGO_BUILD_JOBS and similar are set to the real count (the job's `nproc` lies).
- API calls go through swarmlab's providers (retries, timeouts, caching differ from litellm; the words sent are identical). Tool results reach a swarmlab `llm` agent in a JSON wrapper (only relevant to the optional non-faithful participant).
- The cheating judge is not public; scores are not cheat-filtered.
- Grading: branches share one container instead of one fresh container each.
- Solo shape: one round, one turn (no mid-episode snapshot, so no resume/fork of a half episode).

The prompt sentence reads, for example, `This machine has 8 CPUs and 32 GB of memory.` It is left out when neither number is known.

## The solo shape

A ProgramBench episode is one uninterrupted ReAct loop, so the official solo run is one round with one turn that can take every step:

```yaml
name: programbench-solo
seeds: [0]
budget: {soft_usd: 0, hard_usd: 5.0}
options: {commit: immediate, max_rounds: 1, max_calls_per_turn: 1000}
roles: {solo: {tools: [bash]}}
arms:
  solo:
    world: {type: programbench, params: {task: abishekvashok__cmatrix.5c082c6, backend: docker}}
    participants:
      - {type: llm, count: 1, role: solo,
         params: {model: "anthropic:claude-haiku-4-5", max_tokens: 8192}}
```

The `solo` role leaves the agent `bash` (and `end_turn`, which is always offered) and hides the board's `read_board` and `post`, as in the baseline.

The world needs `commit: immediate`, because a coding agent needs each command's output before its next call, so `validate` refuses `round_end`. At run start, a one-round run must also allow at least `step_limit` tool calls in its turn (`max_calls_per_turn >= 1000`), otherwise the default 20-call cap would end the episode long before the benchmark's limit. Runs with several rounds of k calls are a legitimate variation and are not checked. There the participant enforces the step limit.

The `llm` participant is swarmlab's own loop and does not reproduce the baseline's observation template or step warnings. The faithful participant is mini-swe-agent 2.4.6 itself (next section). `swarmlab job run` without vLLM and `swarmlab programbench grade` come in follow-up PRs.

## Faithful participant

`mini_swe_agent` runs mini-swe-agent 2.4.6, the official ProgramBench baseline agent, as a swarmlab participant. The agent loop is mini-swe-agent's own `DefaultAgent`: its step and wall-clock limits, its step counting, its messages and its templates. swarmlab supplies the model (requests go through `tools.infer`, so the budget, the ledger and the cache apply) and the environment (commands go through the executor to the world's `bash` tool). Module: `swarmlab/participants/minisweagent.py` (its docstring is the exact reference).

```
pip install "swarmlab[programbench]"
```

The participant does not install mini-swe-agent: its agent loop, exceptions and tool-call helpers are a vendored copy of mini-swe-agent 2.4.6 in `swarmlab/vendor/minisweagent/` (MIT; its README lists the files, their wheel hashes and the changes, which are import paths only).

`participants: [{type: mini_swe_agent, params: {...}}]`

| param | default | meaning |
|---|---|---|
| `model` | required | `<provider>:<model id>`, as for `llm` |
| `steps_per_turn` | null | model calls per turn; null runs until the episode ends (the solo shape) |
| `step_limit` | 1000 | the baseline's step limit (model calls), checked before each call; keep it equal to the world's `step_limit` |
| `wall_limit_s` | 21600 | the baseline's wall-clock limit, counted over the agent's own turns |
| `max_tokens` | 8192 | per response (the baseline sets none; swarmlab's providers need one) |
| `temperature` | null | null leaves the provider's default, as the baseline does |
| `extra` | `{}` | provider body fields, as for `llm` |
| `templates` | `solo` | `solo`, `lead`, `worker`, or `file:<path>` (a YAML mapping with `system_template` and `instance_template`) |
| `idle_until_assigned` | false | a worker: no model call until something is delivered; its report to `lead_channel` ends the assignment |
| `wait_tool` | false | a lead: offer `wait_for_workers`, which ends the turn after the step |
| `lead_channel` | `lead` | where workers post reports |
| `worker_channels` | null | the lead's channels to its workers; null: the agent's `post` channels other than `lead_channel` |

Params are in the spec only when they differ from these defaults.

The official solo run is the solo shape of the world with this participant:

```yaml
name: programbench-solo
seeds: [0]
budget: {soft_usd: 0, hard_usd: 5.0}
options: {commit: immediate, max_rounds: 1, max_calls_per_turn: 3000}
roles: {solo: {tools: [bash]}}
arms:
  solo:
    world: {type: programbench, params: {task: abishekvashok__cmatrix.5c082c6, backend: docker}}
    participants:
      - {type: mini_swe_agent, count: 1, role: solo, params: {model: "anthropic:claude-haiku-4-5"}}
```

`max_calls_per_turn` counts tool calls, not steps, and one reply may run several commands, so it is set well above the 1000-step limit; the run-start check only requires at least `step_limit`. The executor's cap is a safety net here: the episode should end by submission or by one of the baseline's limits.

How the transcript matches the baseline:

- The system message is the baseline's system template and the first user message its instance template, rendered with the same machine details; the world's resource sentence follows it (the one added sentence).
- Each reply is stored as the model's assistant message with its tool calls. Each `bash` result comes back as a tool message rendered with the baseline's `observation_template`: the return code, the output (head and tail past 10000 characters), and the warnings when fewer than 20 steps or less than 10 minutes are left.
- A reply without a tool call, with bad `bash` arguments or calling a tool that is not offered gets the baseline's `format_error_template` message (the "output token limit" variant when the reply was cut off); three in a row end the episode (`RepeatedFormatError`).
- A submission, the step limit and the wall limit end the episode with the baseline's exit messages (`Submitted`, `LimitsExceeded`, `TimeExceeded`). The workspace is graded as it is in every case.
- Differences: swarmlab always offers `end_turn` next to `bash`; a model that calls it ends its turn, which in the solo shape ends the run with the workspace as it is. Requests go through swarmlab's providers (see the deviations above). The participant's whole conversation, mini-swe-agent's message dicts, is in its snapshot state (`messages`), and the model calls are in the event log as for any participant.

## Swarm shape

The lead/worker scaffold of arXiv 2610.17471 with this participant: a lead with the official task text and delegation instructions, workers that see only their assignments, a shared workspace, and k steps per turn:

```yaml
name: programbench-team
seeds: [0]
budget: {soft_usd: 0, hard_usd: 20.0}
options: {commit: immediate, max_rounds: 60, max_calls_per_turn: 40, scheduler: {type: fixed}}
roles:
  lead: {tools: [bash, post, read_board], channels_write: [worker-1, worker-2, worker-3]}
  worker: {tools: [bash, post, read_board], channels_read: [worker-1, worker-2, worker-3]}
arms:
  team:
    world: {type: programbench, params: {task: abishekvashok__cmatrix.5c082c6, backend: docker}}
    medium:
      delivery: pull
      topology:
        type: rooms
        params:
          rooms:
            lead: [a000, a001, a002, a003]
            worker-1: [a000, a001]
            worker-2: [a000, a002]
            worker-3: [a000, a003]
    participants:
      - {type: mini_swe_agent, count: 1, role: lead,
         params: {model: "anthropic:claude-haiku-4-5", templates: lead, steps_per_turn: 10,
                  wait_tool: true}}
      - {type: mini_swe_agent, count: 3, role: worker,
         params: {model: "anthropic:claude-haiku-4-5", templates: worker, steps_per_turn: 10,
                  idle_until_assigned: true}}
```

- Channels: one room per worker, shared with the lead (`worker-1`: the lead `a000` and the worker `a001`), carries its assignments; the `lead` room carries the reports. The lead's role writes only the worker rooms; a worker's role reads only worker rooms, so of its rooms it reads its own and never the other workers' reports. The `tree` topology does not fit: its channels are the groups and one `coordinators` channel, so a lead cannot have a channel of its own with each worker.
- Delegation is a post on the worker's room; the lead's prompt names the rooms (`worker_channels`). A report is a post on `lead`, starting with the worker's room name, since board messages carry no author.
- Pull delivery: nobody's model sees a message unless it calls `read_board`. Each idle worker (`idle_until_assigned`) still reads its board once per turn itself, with no model call; once something arrives it works for up to 10 steps per turn until it posts its report, then it is idle again. Its next assignment continues the same conversation. Under `delivery: push` the deliveries are inserted as one user message at the start of the turn instead (use `push_consume: true` so each is pushed once).
- `wait_for_workers` ends the lead's turn after the current step; the reports are on the board at its next turn.
- The `fixed` scheduler gives the lead the first turn of every round, so with immediate commit an assignment reaches its worker in the same round.
- Only the lead submits. The worker template tells workers never to run the submission command, which would end the run for the whole team.
- Each agent has its own step limit and its own clock (its turns only); the world's 6 h clock is shared, and past it every command is refused, which ends each agent's episode (`TimeExceeded`).

Check a spec without spending: `swarmlab validate <spec>` and `swarmlab prompts <spec>` show the exact system and first user message of the lead and of a worker.

## Snapshots, resume and the viewer

Each round's world snapshot holds the run state plus a deterministic `.tar.gz` of the whole workspace (minus `snapshot_exclude`). An unchanged workspace gives the same bytes, so the blob store keeps one copy. A snapshot made before the docker backend ever started holds no tar, meaning the image's untouched workspace. The grade step will read the last snapshot's tar as `submission.tar.gz`. Snapshots contain the reference binary and docs, whose upstream licences include the GPL, so keep run directories private.

Restoring a snapshot touches no container and no file. The viewer's "World state" panel shows the workspace's top-level entries with sizes, the step count, whether and by whom the task was submitted, and the last 10 commands with their return codes. Resume and fork extract the tar into the workspace at the start of the next round, in a fresh container under Docker. Running processes (a tmux session, a background server) do not survive a snapshot boundary. In the solo shape there is no boundary inside the episode.

`score()` is cheap and runs every round: `submitted`, `steps` (bash commands), `elapsed_s` (the world's clock, which starts at the first round and does not count backend start-up), and `graded: false` until the grade step exists. Each round's `round_resolved` event records the steps and the submission.

## Dry run (free)

`fake:programbench_echo` runs `ls`, writes a trivial `compile.sh` that builds `./executable`, runs it and submits. It needs no Docker, image or key:

```yaml
name: programbench-dry
budget: {soft_usd: 0, hard_usd: 0}
options: {commit: immediate, max_rounds: 1, max_calls_per_turn: 5}
arms:
  echo:
    world:
      type: programbench
      params: {task: abishekvashok__cmatrix.5c082c6, workspace: /tmp/pb-dry, user: null,
               lockdown: false, check_task: false, step_limit: 5}
    participants: [{type: llm, count: 1, params: {model: "fake:programbench_echo", max_tokens: 256}}]
```

The local backend writes into `workspace` and empties it when a snapshot is restored there, so point a local run at a scratch directory, never at a checkout you care about.

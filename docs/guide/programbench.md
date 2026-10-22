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
| `pin_hosts` | `[api.anthropic.com, huggingface.co, hf.co, cas-server-direct.xethub.hf.co, cas-server.xethub.hf.co]` | the hosts that stay reachable under lockdown: the Anthropic API, the Hub, and the Xet storage hosts the job's bucket sync uploads to |
| `max_output_bytes` | 1000000 | safety cap on one command's output (head and tail kept around one elision line); `null` disables it |
| `cpus`, `memory_gb` | null | the resources the task text states; docker defaults to the official 20 CPUs and 60 GB, the local backend detects them from the cgroup under lockdown |
| `snapshot_exclude` | `[]` | fnmatch patterns left out of the workspace snapshot (e.g. `target`); by default the whole workspace is kept, since it is the submission |
| `check_task` | true | at run start, refuse a task the `programbench` package does not know |

Every param is in the spec and its hash. That includes `backend`: a run inside the task image and a Docker run are different experiments.

## The two backends

- **`local`** runs `bash -lc <command>` directly, in its own session, with the output going to a temp file. This is the shape for an HF Job that runs swarmlab inside the task image (Jobs cannot run Docker). The command environment is the inherited one minus anything secret-looking (`ANTHROPIC_*`, `HF_TOKEN`, `*_KEY`, `*_TOKEN`, `*_SECRET`, ...), minus every name the job launcher lists in `SWARMLAB_SECRET_NAMES` (so a key in a variable like `MODEL_AUTH` is dropped too) and minus swarmlab's own virtualenv, plus the baseline variables and, with a known CPU count, `MAKEFLAGS=-j<n>`, `CARGO_BUILD_JOBS`, `OMP_NUM_THREADS` and `GOMAXPROCS`.
- **`docker`** runs `docker run -d --network none --cpus 20 --memory 60g --memory-swap 60g --user agent --cap-drop SYS_PTRACE <image> sleep 7h` at the first command, then `docker exec -w /workspace -e PAGER=cat ... bash -lc <command>` per call, as mini-swe-agent does. The container is removed at the end of the last round or tick (terminal, or the run's `max_rounds`), after every command still running in it is killed; a run that stops another way (budget, stop condition, crash) relies on a cleanup hook that runs at interpreter exit.

Neither backend starts before the first command: `swarmlab prompts`, `replay`, `report` and the viewer never start a container or lock a machine down.

Under lockdown (local backend, `lockdown: true`, swarmlab running as root), the first command first pins each `pin_hosts` host in `/etc/hosts`, points `/etc/resolv.conf` at `0.0.0.0`, and removes the agent user's sudo rules and the sudo binary's permissions. Nothing happens when swarmlab is not root.

## Official semantics kept

- One tool, `bash(command)`, with mini-swe-agent's schema. Its feedback is the baseline's result dict, `{"output", "returncode", "exception_info"}`: stdout and stderr merged, never truncated for the agent (the 5000/5000-character head and tail rendering past 10000 characters is the participant's job). The only cap is `max_output_bytes`.
- Every call is a fresh `bash -lc` in the workspace, so `cd` and `export` do not persist. The environment carries `PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1`.
- On a timeout the agent sees the output so far, return code -1 and the baseline's `An error occurred while executing the command: ...` text. The process keeps running, as under Docker, where only the waiting client is killed. The world keeps track of it until it exits; a submission or the run's end kills it.
- A command whose first non-blank output line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, with return code 0, submits. The run is then terminal and any later `bash` is refused (`submitted`). There is no `submit` tool. At a limit, the workspace is submitted as it is.
- Submission freezes the workspace: when a command submits, the world kills every other command still running (another agent's, under `commit: async`, or a timed-out one in any mode) and the background jobs earlier commands left behind, so the snapshot that ends the run holds the workspace the submission saw. A killed command returns the usual three keys: its output so far, return code `-9`, and `exception_info` `An error occurred while executing the command: Command '<command>' was killed: a000 submitted the final output (the episode is over)`. The run's end, whatever ends it (the submission, `max_rounds`, an agent's `rounds_per_agent`, a budget, a stop condition), kills whatever is still running (`the run ended`) the same way, before the final snapshot.
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

The world needs `commit: immediate` or `commit: async`, because a coding agent needs each command's output before its next call, so `validate` refuses `round_end`. The official solo shape is `immediate`, one round, one turn; `async` is for teams ([Async team](#async-team)). At run start, a one-round run must also allow at least `step_limit` tool calls in its turn (`max_calls_per_turn >= 1000`), otherwise the default 20-call cap would end the episode long before the benchmark's limit. Runs with several rounds of k calls are a legitimate variation and are not checked. There the participant enforces the step limit.

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
| `idle_until_assigned` | false | a worker: no model call until something is delivered; its report to `lead_channel` ends the assignment. Under `commit: async` it waits for the delivery with the core `wait` tool (its role must list `wait`) |
| `wait_tool` | false | a lead: offer `wait_for_workers`, which ends the turn after the step; under `commit: async` it waits for the reports and returns them instead (its role must list `wait`). Not with `templates: solo` |
| `lead_channel` | `lead` | where workers post reports |
| `worker_channels` | null | the lead's channels to its workers; null: the agent's `post` channels other than `lead_channel` |
| `guidance` | null | a paragraph appended to the instance text (the paper's "guidance", e.g. "Delegate the implementation to your workers"): the literal text, or `file:<path>`, a text file read at run time on the machine that runs the spec (leading `# ` lines are a header comment and are not sent). With `templates: solo` the run is no longer the official one: it is the paper's "Solo PIV"-style run, whose prompts are the official ones plus exactly that paragraph (before the resource sentence) |

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

- The model is offered exactly the baseline's one tool, mini-swe-agent's `BASH_TOOL` (`bash`, "Execute a bash command", one `command` string), and nothing else: not `end_turn`, which swarmlab's executor lists for every agent, nor the board's tools. `swarmlab prompts` shows `tools: bash`.
- The system message is the baseline's system template and the first user message its instance template, rendered with the same machine details; the world's resource sentence follows it (the one added sentence).
- Each reply is stored as the model's assistant message with its tool calls. Each `bash` result comes back as a tool message rendered with the baseline's `observation_template`: the return code, the output (head and tail past 10000 characters), and the warnings when fewer than 20 steps or less than 10 minutes are left.
- A reply without a tool call, with bad `bash` arguments or calling any other tool (`end_turn` included) gets the baseline's `format_error_template` message (the "output token limit" variant when the reply was cut off) and counts as a step; the episode goes on, and three in a row end it (`RepeatedFormatError`).
- A submission, the step limit and the wall limit end the episode with the baseline's exit messages (`Submitted`, `LimitsExceeded`, `TimeExceeded`). The workspace is graded as it is in every case.
- The turn ends only when the episode does (submission, step limit, wall limit, repeated format errors, a context-length error) or when swarmlab's own limits stop it (the executor's `max_calls_per_turn`, the budget's hard ceiling).
- Differences: requests go through swarmlab's providers (see the deviations above). Their adapters send a tool schema exactly as the participant offers it, so the wire schema is the baseline's: `BASH_TOOL` without `additionalProperties`, and on Anthropic without `strict`. The participant's whole conversation, mini-swe-agent's message dicts, is in its snapshot state (`messages`), and the model calls are in the event log as for any participant.
- Differences: a context-length error from the provider (HTTP 400 naming the context length, e.g. vLLM's "maximum context length") ends the episode with the exit message `ContextExceeded`, as the step limit does, and the run ends normally. The baseline carries the whole transcript with no context management too, and would crash on that error instead. The workspace is graded as it is either way.

## Swarm shape

The lead/worker scaffold of arXiv 2610.17471 with this participant: a lead with the official task text and delegation instructions, workers that see only their assignments, a shared workspace, and k steps per turn. This section is the lockstep variant (`commit: immediate`, rounds, the `fixed` scheduler): every agent takes one turn per round, so a fast agent waits for the slowest one every round. The [async team](#async-team) below runs the same team with no round barrier.

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

## Async team

The same team under `commit: async` (`examples/programbench_team_async.yaml`): there are no rounds and no barrier, every agent works at its own pace, and messages arrive while an agent works, not at the next round.

```yaml
name: programbench-team-async
seeds: [0]
budget: {hard_usd: 5.0}
options: {commit: async, rounds_per_agent: 30, max_calls_per_turn: 40}
roles:
  lead: {tools: [bash, post, read_board, wait], channels_write: [worker-1, worker-2]}
  worker: {tools: [bash, post, read_board, wait], channels_read: [worker-1, worker-2]}
arms:
  team:
    world: {type: programbench, params: {task: abishekvashok__cmatrix.5c082c6}}
    medium:
      delivery: pull
      topology:
        type: rooms
        params:
          rooms:
            lead: [a000, a001, a002]
            worker-1: [a000, a001]
            worker-2: [a000, a002]
    participants:
      - {type: mini_swe_agent, count: 1, role: lead,
         params: {model: "anthropic:claude-haiku-5-5", templates: lead, steps_per_turn: 10,
                  step_limit: 100, wait_tool: true}}
      - {type: mini_swe_agent, count: 2, role: worker,
         params: {model: "anthropic:claude-haiku-5-5", templates: worker, steps_per_turn: 10,
                  step_limit: 100, idle_until_assigned: true}}
```

The world's `backend` is left at `local`, the HF Jobs shape (`swarmlab job run` plans it); set `backend: docker` to run it on a machine with Docker. It needs the task image either way, so fake models cannot run it; `swarmlab validate` and `swarmlab prompts` work offline.

- **Ticks.** Each agent loops over its own turns, all at once; every completed turn of any agent closes one tick, which is what events, snapshots and `max_rounds` count. `rounds_per_agent: 30` caps every agent at 30 turns (so at most 90 ticks). There is no scheduler: `commit: async` refuses one.
- **`steps_per_turn` is the tick cadence.** A turn is at most 10 model calls, so each agent's progress is logged and can be snapshotted at least every 10 steps. It no longer makes anyone wait: a worker's 10 steps and the lead's 10 steps run at the same time, and their commands run at the same time in the shared workspace.
- **The `wait` primitive.** Both roles list the core `wait` tool (offered only under async). The participant uses it; its model never sees it. An idle worker calls it itself, with no model call: its turn parks until the lead posts on its room, the post becomes its assignment, and it works up to 10 steps per turn until it posts its report on `lead`, then waits again. The lead's `wait_for_workers` is one `wait`: it returns the reports as its tool result (rendered as a `read_board` result) and the lead's turn goes on. The lead's prompt says so ("it returns them as its result"); the worker's prompt says that between assignments it waits.
- **The deadlock rule.** A `wait` returns at once, empty, when every other running agent is waiting or finished, since nobody could post. The lead then reads `no worker can report: every worker is idle or finished ...` and goes on. An idle worker's turn ends with the note `idle` and its next turn waits again; such a turn costs no model call but closes a tick, so a worker that is never given work ends at its `rounds_per_agent` cap (or when the run ends).
- **The submission freeze.** When the lead's submission command returns, the world kills every command still running (the workers' included) and refuses new ones. The run ends at the tick the lead's turn closes, so its last turn is in the log; the other agents' turns still in flight are then cancelled, and their commits stand.
- **The final snapshot.** The tick that ends the run is always snapshotted, whatever `snapshot_every`, so grading reads the frozen workspace. Snapshots of earlier ticks may catch a file another agent's command is writing; resume and fork lose the turns that were in flight anyway.
- **Clocks.** Each agent's own 6 h clock (the participant's `wall_limit_s`) is plain wall time inside its turns, including the time it spends in `wait`. The world's shared clock (the world's `wall_limit_s`, from the first tick) is the real limit: past it every command is refused, which ends every agent's episode (`TimeExceeded`).
- **Step limits.** `step_limit: 100` is per agent (each participant's own limit, at most 300 model calls for the team). The world's `step_limit` is only checked for a one-round (or one-tick) run.
- The solo official shape stays `commit: immediate`, one round, one turn.

## Snapshots, resume and the viewer

Each round's world snapshot holds the run state plus a deterministic `.tar.gz` of the whole workspace (minus `snapshot_exclude`). An unchanged workspace gives the same bytes, so the blob store keeps one copy. A snapshot made before the docker backend ever started holds no tar, meaning the image's untouched workspace. The grade step will read the last snapshot's tar as `submission.tar.gz`. Snapshots contain the reference binary and docs, whose upstream licences include the GPL, so keep run directories private.

Restoring a snapshot touches no container and no file. The viewer's "World state" panel shows the workspace's top-level entries with sizes, the step count, whether and by whom the task was submitted, and the last 10 commands with their return codes. Resume and fork extract the tar into the workspace at the start of the next round, in a fresh container under Docker. Running processes (a tmux session, a background server) do not survive a snapshot boundary. In the solo shape there is no boundary inside the episode.

`score()` is cheap and runs every round: `submitted`, `steps` (bash commands), `elapsed_s` (the world's clock, which starts at the first round and does not count backend start-up), and `graded: false` until `swarmlab programbench grade` writes the score ([Grading](#grading)). Each round's `round_resolved` event records the steps and the submission.

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

## Running in HF Jobs

HF Jobs cannot run Docker, so a ProgramBench run executes swarmlab inside the task image itself, with the `local` backend. `swarmlab job run` does this when no arm calls a `vllm:` model and `--model` is left out (API mode: no vLLM is installed or served).

```
swarmlab job run solo.yaml                 # prints the plan and the exact hf jobs command
swarmlab job run solo.yaml --launch        # uploads the stage dir and submits
```

Use the solo YAML above with `backend: local` (the default) and an API model, e.g. `model: "anthropic:claude-haiku-5-5"`.

- **Image**: the task image `programbench/<task>:task_cleanroom_v6` (`--image` overrides it). The planner refuses a world whose `backend` is not `local` (it does not rewrite the spec), arms that name different tasks (one job, one image), and more than one programbench run per job. Each run works in the image's own `/workspace`, which nothing resets between runs, so launch one job per arm and seed (`--arm`, `--seeds`).
- **Flavor and timeout**: `cpu-upgrade` (8 vCPU, 32 GB, $0.03/h) by default; `--flavor cpu-xl` or `cpu-performance` for big builds. The timeout defaults to the world's `wall_limit_s` plus 1 h for setup, the final snapshot and the sync, which is 7 h for the official 6 h limit.
- **Secrets**: `HF_TOKEN` (stage download and the bucket sync) and the model's key variable are passed by name, `--secrets HF_TOKEN --secrets ANTHROPIC_API_KEY`, so the hf CLI reads each value from your shell. If only `ANTHROPIC_KEY` is set, that name is passed instead (the provider reads either). The names, never the values, are recorded in `plan.json`, the launch ledger and the job's `job.json`. `--launch` refuses while one is unset. The agent's commands never see any of them: the world strips secret-looking variables and every name in `SWARMLAB_SECRET_NAMES` (the exact `--secrets` names, including a custom `api_key_env`) from their environment, and the grader does the same.
- **Bootstrap**: the image has Python 3.10, pip and curl, but no uv. The job's command installs uv into `/opt/swarmlab/uv`, fetches the stage dir, then `bootstrap.sh` (mode `api`) installs a managed Python 3.12 and `swarmlab[anthropic,programbench]` from the staged wheel. It then runs `swarmlab run` and syncs the run dir to the bucket. `/opt/swarmlab` (venv, stage, run dirs, logs) is made root-only, so the agent user cannot read it.
- **Lockdown in the job**: the job runs as root, so at the world's first `bash` command (not at job start) the world pins `pin_hosts` in `/etc/hosts`, blackholes DNS and removes the agent's sudo, then runs every command as `agent`. The pinned hosts include the Hub and the Xet storage hosts, because the bootstrap's bucket sync keeps running after the lockdown.
- **Hidden tests**: none are in this job. No dataset is mounted; the tests reach only the grade job below.

What to expect in the job log: `mode=api tag=... job=...`, `installing uv`, `installing a managed Python 3.12`, `installing swarmlab-...whl [anthropic,programbench]`, `run arm=solo seed=0`, then little until the run ends: the run's estimate and status lines, while the agent's commands go to the run's events, not the log. Finally `finishing: status=completed rc=0; final sync ...` and `JOB_EXIT 0`. Bring the run home with `swarmlab job fetch <run_id> --tag <tag>`. `swarmlab view`, `report` and `export` then work on it, and the score says `graded: false`.

**A self-served model.** To run on an open model that needs a GPU, such as Flint, serve it in a second job and point the run at it: `swarmlab job serve RecursiveSI/flint --launch`, then, once `swarmlab job endpoint <id>` says ready, `swarmlab job run examples/programbench_team_async_flint.yaml --endpoint <id> --launch`. The run job stays a CPU job in the task image; the lockdown also pins the endpoint's host. See [Self-served models in HF Jobs](real-models.md#self-served-models-in-hf-jobs) for the provider block, the cost and how to stop the server.

**Files the spec references.** A participant param `file:<path>` (`templates: file:...`, `guidance: file:...`) is copied into the stage dir under `files/` and the job's spec points at the copy; two different files with the same name are refused. The run's spec hash depends on the file's content, not its path, so the job's runs hash like local ones.

## Grading

```
swarmlab programbench grade runs/<run_id>                    # prints the plan and the command
swarmlab programbench grade runs/<run_id> --launch [--wait]  # submits; --wait polls, then writes the score
swarmlab programbench fetch-grade runs/<run_id>              # later: download grade.json, write the score
```

`grade` writes the run's last committed snapshot workspace to `runs/<run_id>/artifacts/submission.tar.gz`. If that snapshot holds no workspace (no command ever ran), it writes an empty `artifacts/submission.untouched` marker and refuses, since there is nothing of the agent's to grade. It also refuses when the last snapshot is older than the run's last round, naming both rounds: the final workspace was never saved. The round (or tick) that ends a run is always snapshotted, whatever `snapshot_every`, so this happens only to a run that did not end normally (a hard ceiling or a crash discards the round in flight); resume it to finish it. It then plans a second job:

- image: the task's clean `task_cleanroom_v6` image, which is also what the official evaluator uses;
- flavor `cpu-upgrade` by default. pytest runs with as many workers as the job's real CPU count, read from the cgroup;
- timeout: the evaluator's bounds (30 min setup + 900 s compile + 3600 s per active test branch), capped at 12 h. cmatrix has 14 branches, so its bound is 14.8 h and the default is 12 h. A small task actually grades in minutes. `--timeout` overrides it;
- the hidden tests: `-v hf://datasets/programbench/ProgramBench-Tests:/root/programbench-tests:ro`, mounted only in this job and under `/root`. They contain the task's original source;
- `--secrets HF_TOKEN` only.

Inside the job, `swarmlab/jobs/grade.py` runs ProgramBench 1.2.5's own `Evaluator`, filled with the arguments its batch runner uses. Only its container class is swapped for a local one, because the job's own filesystem is the container. Every evaluator step therefore runs unchanged:

- the workspace wipe, the submission extraction and the removal of gold-hash files and `./executable`;
- the seeded git repo, and `compile.sh` with DNS blackholed under 900 s;
- the stashed and hashed binary and the `pytest-rerunfailures` install;
- per branch: the tests archive, `eval/run.sh` with `--reruns=2` under 3600 s, the serial retry after xdist worker crashes, and the JUnit XML parsing.

The score is `programbench.submission`'s: the fraction of non-ignored tests on the active branches that pass, `resolved` at 1.0 and `near_resolved` at 0.95. The job uploads `grade.json` (the full `EvaluationResult`, the score, CPUs, image, programbench version) to its stage dir.

The submission's code runs as `agent`, not root. The official evaluator runs everything as root, but in a compile container with no dataset mount, and later copies one branch's tests into each test container. Here the hidden tests and the grader share the job, so the three evaluator commands that execute submitted code (the git seed, `compile.sh` and `eval/run.sh`) drop to `agent` with `setpriv --no-new-privs`. With no new privileges, the image's sudo for `agent` cannot elevate. Every other evaluator step still runs as root. The tests under `/root` (0700) and the grader under `/opt/swarmlab` (0700) are out of `agent`'s reach, and the grader checks that before it starts. At every teardown it kills every process `agent` left running, including `setsid` daemons. One cost: a `compile.sh` that needs root, or a test whose expected output depends on running as root, can fail here and pass officially.

The other deviation: the official evaluator starts a fresh container from the post-compile image for each branch. Here the branches share the job, and `/workspace` and the stashed binary are reset from the post-compile tar before each branch. Files a test leaves elsewhere (`/tmp`, `agent`'s `$HOME`) carry over to later branches. Both deviations are recorded in `grade.json`.

Where the score lands: `fetch-grade` (or `grade --wait`) keeps `grade.json` in `runs/<run_id>/artifacts/` and rewrites `run.json`. An export made before that is no longer current, so the next `publish` re-exports with the grade. `score` becomes the world's score plus `graded: true, score, passed, total, resolved, near_resolved, branches, compile_ok`. `score_world` keeps the runner's original value, and `grade` records the job, versions and deviation. This is the one documented exception to "the runner owns the score": replay checks the world's own score against `score_world`, so `Run.load`, `swarmlab report` (its Final score table shows the graded keys) and `export` work on a graded run. A re-grade replaces the graded keys.

Scores are not cheat-filtered: the benchmark's automated cheating judge is not public. Read the transcript (`swarmlab view`, `export`) before trusting a high score.

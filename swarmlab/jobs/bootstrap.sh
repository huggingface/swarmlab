#!/usr/bin/env bash
# swarmlab HF Job bootstrap (docs/handoff/WP8.md). Runs INSIDE the job, started by the command
# `swarmlab job run` (or `swarmlab programbench grade`) prints: fetch the stage dir from the
# bucket, then `bash bootstrap.sh`.
#
# SWARMLAB_MODE selects one of three branches (default vllm, the original behaviour):
#   vllm   install swarmlab, install and serve vLLM, probe it, then the runs (below)
#   api    API models, no vLLM (launch.py "API mode"): uv and a managed Python 3.12 are installed
#          first when the image lacks them (the ProgramBench task image has Python 3.10, pip and
#          curl, no uv), then swarmlab, then the runs; nothing vLLM (no install, serve, probe or
#          argv/port agreement checks). SWARMLAB_UV is the uv the job entry installed or found.
#   grade  `swarmlab programbench grade`: install as in api, then `python -m swarmlab.jobs.grade`
#          on $STAGE_DIR/submission.tar.gz with the tests mounted at SWARMLAB_GRADE_TESTS; the
#          result is uploaded as $SWARMLAB_STAGE/grade.json (no runs, no run sync).
#   api and grade: ROOT is made root-only (chmod 700) and the scratch dir defaults to $ROOT/tmp,
#   so the agent user of a ProgramBench task image cannot read run dirs or job logs.
#   SWARMLAB_EXTRAS (comma-separated, e.g. anthropic,programbench) installs the wheel with those
#   extras; SWARMLAB_SECRET_NAMES (names only) is recorded in job.json.
#   SWARMLAB_ENDPOINT_URL / SWARMLAB_ENDPOINT_JOB (api, `swarmlab job run --endpoint`): the base URL
#   and job id of a model served by another job, recorded in job.json `endpoint` (the providers
#   read their own `base_url_env`; the programbench world reads SWARMLAB_PIN_HOSTS_EXTRA).
# A serving job (`swarmlab job serve`) does not run this script: it runs serve.sh (same stage-dir
# fetch, same exit-code table below, plus 15: the server exited after it was ready).
#
# Every setting comes from the environment (set by swarmlab/jobs/launch.py with `-e`):
#   SWARMLAB_TAG          job tag, e.g. m2-vllm-qwen9b-1a2b3c4-20261006-120000
#   SWARMLAB_STAGE        hf://buckets/<ns>/<bucket>/jobs/<tag>  (wheel, spec.yaml, this script)
#   SWARMLAB_RUNS_REMOTE  hf://buckets/<ns>/<bucket>/runs/<tag>  (run dirs are synced here)
#   SWARMLAB_RUNS         space-separated "arm:seed" pairs, started in this order
#   SWARMLAB_PARALLEL     runs at once (default 1: one after another), vllm and api modes alike
#                         (vllm: against the one server); each is its own `swarmlab run` process
#                         with its own joblogs/run-* files, and results.jsonl gets its line when
#                         it ends (completion order). >1 needs bash >= 5.1 (`wait -n -p`). The
#                         launcher never sends it with a programbench run (one run per job there)
#   SWARMLAB_GIT_COMMIT / SWARMLAB_GIT_DIRTY / SWARMLAB_JOB_SPEC_SHA256 / SWARMLAB_FLAVOR / SWARMLAB_IMAGE
#   SWARMLAB_SYNC_EVERY   seconds between background syncs of $SWARMLAB_JOB_TMP/runs (default 300)
#   VLLM_MODEL VLLM_PORT VLLM_VERSION VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION VLLM_FLAGS
#   The `vllm serve` argv is NOT rebuilt from these: it is read from $STAGE_DIR/plan.json
#   (`vllm_serve_argv`, the list `swarmlab job run` printed as `vllm_serve_command`) and passed
#   NUL-delimited, so a quoted token with spaces or newlines (a JSON value, an inline chat
#   template) stays one argument and the executed command is the planned one. The env vars are
#   recorded in job.json and must agree with that argv (model, port, VLLM_FLAGS as its shell-quoted
#   tail), else exit 12. The launcher's flags (--served-model-name --host --port --max-model-len
#   --gpu-memory-utilization, --revision) come first in the argv; VLLM_FLAGS never repeats them.
#   VLLM_FLAGS_SOURCE / VLLM_PROFILE (recorded in job.json)
#   VLLM_PROBE_EXTRA (JSON object merged into the readiness probe's request; `{}` sends none; unset
#   or empty: chat_template_kwargs.enable_thinking=false) and VLLM_PROBE_EXTRA_SOURCE (where the
#   launcher took it from: --probe-extra | profile | default | none); both recorded in job.json
#   VLLM_REVISION (optional)  VLLM_WAIT_S (default 1200: /health + /v1/models must answer by then)
#
# Layout: vLLM in the image's system Python (`uv pip install --system`, as in the textworld
# study: the server must not share a venv with swarmlab); swarmlab + the hf CLI in their own
# venv; runs on local disk (/tmp/runs by default: bucket mounts drop exec bits and refuse sqlite/sockets),
# mirrored to the bucket with `hf buckets sync` after every run and every SWARMLAB_SYNC_EVERY s.
# job.json (manifest), results.jsonl (one CLI summary per run), vllm.log and job-side logs land
# in SWARMLAB_STAGE. Exit codes: 0 all runs ran (grade: grade.json written); 1 a run failed;
# 10 install (or a bad SWARMLAB_PARALLEL, or > 1 under bash < 5.1); 11 vLLM install; 12 vLLM not
# ready (or plan.json's argv missing / disagreeing with the env); 13 endpoint probe failed (HTTP
# error); 14 grading failed (no grade.json); 143 SIGTERM (job timeout or cancel).
# A probe answer without a parsed tool call only warns: grep the log for SWARMLAB_PROBE_WARNING;
# job.json "probe" records the outcome (ok | no_tool_call | error).
set -uo pipefail

T0=$(date +%s)
log() { echo "[swarmlab-job +$(( $(date +%s) - T0 ))s] $*"; }
MODE="${SWARMLAB_MODE:-vllm}"
case "$MODE" in vllm|api|grade) ;; *) echo "unknown SWARMLAB_MODE=$MODE (vllm|api|grade)" >&2; exit 10 ;; esac
: "${SWARMLAB_TAG:?}" "${SWARMLAB_STAGE:?}"
if [ "$MODE" != grade ]; then : "${SWARMLAB_RUNS_REMOTE:?}" "${SWARMLAB_RUNS:?}"; fi
if [ "$MODE" = vllm ]; then : "${VLLM_MODEL:?}"; fi
if [ "$MODE" = grade ]; then : "${SWARMLAB_GRADE_TASK:?}" "${SWARMLAB_GRADE_TESTS:?}"; fi
ROOT="${SWARMLAB_JOB_ROOT:-/opt/swarmlab}"   # venv + work dir
if [ "$MODE" = vllm ]; then
  JT="${SWARMLAB_JOB_TMP:-/tmp}"             # local-disk scratch: runs, logs, manifest
else
  JT="${SWARMLAB_JOB_TMP:-$ROOT/tmp}"        # root-only: the agent user must not read it
fi
STAGE_DIR="${SWARMLAB_STAGE_DIR:-$ROOT/stage}"
VENV="$ROOT/venv"
WORK="$ROOT/work"
OUT="$JT/runs"
SYNC_EVERY="${SWARMLAB_SYNC_EVERY:-300}"
PARALLEL="${SWARMLAB_PARALLEL:-1}"
VLLM_WAIT_S="${VLLM_WAIT_S:-1200}"
VLLM_PORT="${VLLM_PORT:-8000}"   # the spec's providers override points at this port
BASE="http://127.0.0.1:$VLLM_PORT"
export HF_HOME="${HF_HOME:-$JT/hf}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ "$MODE" = vllm ]; then
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  export SWARMLAB_OFFLINE=1   # never fetch the router catalog inside the job
fi
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STATUS=starting RC=0 VLLM_PID="" SYNC_PID="" READY_S="" FAILED_RUNS="" DONE_RUNS=""
declare -A RUNNING=()   # pid -> "arm:seed" of the `swarmlab run` processes in flight
VLLM_CMDLINE=""
if [ -z "${VLLM_PROBE_EXTRA:-}" ]; then   # launched by hand: the launcher's default
  VLLM_PROBE_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  VLLM_PROBE_EXTRA_SOURCE="${VLLM_PROBE_EXTRA_SOURCE:-default}"
fi
export VLLM_PROBE_EXTRA VLLM_PROBE_EXTRA_SOURCE="${VLLM_PROBE_EXTRA_SOURCE:-}"
HF="hf"
UV="${SWARMLAB_UV:-uv}"
uv() { command "$UV" "$@"; }   # the image's uv (vllm), or the one the entry installed
mkdir -p "$OUT" "$WORK" "$JT/joblogs"
[ "$MODE" = vllm ] || chmod 700 "$ROOT"

hf_q() { "$HF" "$@" >>"$JT/joblogs/hf.log" 2>&1; }

# ---- manifest ---------------------------------------------------------------------------------
write_manifest() {
  ENDED_AT="${1:-}" STARTED_AT="$STARTED_AT" STATUS="$STATUS" RC="$RC" READY_S="$READY_S" \
    DONE_RUNS="$DONE_RUNS" FAILED_RUNS="$FAILED_RUNS" VLLM_CMDLINE="$VLLM_CMDLINE" MODE="$MODE" \
    PROBE_FILE="$JT/probe.json" GRADE_FILE="$JT/grade.json" python3 - <<'PY' >"$JT/job.json"
import json, os
e = os.environ.get
try:
    probe = json.load(open(e("PROBE_FILE")))
except (OSError, ValueError):
    probe = None
try:
    probe_extra = json.loads(e("VLLM_PROBE_EXTRA") or "")
except ValueError:
    probe_extra = e("VLLM_PROBE_EXTRA")   # unparsable: recorded as given (the probe fails on it)
grade = None
if e("MODE") == "grade":
    try:
        g = json.load(open(e("GRADE_FILE")))
        grade = {"task": g.get("task"), "score": g.get("score")}
    except (OSError, ValueError):
        grade = {"task": e("SWARMLAB_GRADE_TASK"), "score": None}
doc = {
    "mode": e("MODE"), "secrets": (e("SWARMLAB_SECRET_NAMES") or "").split(),
    "extras": e("SWARMLAB_EXTRAS") or None,
    "tag": e("SWARMLAB_TAG"), "job_id": e("JOB_ID"), "flavor": e("SWARMLAB_FLAVOR"),
    "accelerator": e("ACCELERATOR"), "image": e("SWARMLAB_IMAGE"),
    "commit": e("SWARMLAB_GIT_COMMIT"), "dirty": e("SWARMLAB_GIT_DIRTY") == "1",
    "job_spec_sha256": e("SWARMLAB_JOB_SPEC_SHA256"),
    "started_at": e("STARTED_AT"), "ended_at": e("ENDED_AT") or None,
    "status": e("STATUS"), "exit_code": int(e("RC") or 0),
    "runs_planned": (e("SWARMLAB_RUNS") or "").split(),
    "runs_done": (e("DONE_RUNS") or "").split(), "runs_failed": (e("FAILED_RUNS") or "").split(),
    "runs_remote": e("SWARMLAB_RUNS_REMOTE"), "parallel": int(e("SWARMLAB_PARALLEL") or 1),
    "vllm": None if e("MODE") != "vllm" else {
             "model": e("VLLM_MODEL"), "revision": e("VLLM_REVISION") or None,
             "version": e("VLLM_VERSION"), "max_model_len": e("VLLM_MAX_MODEL_LEN"),
             "gpu_memory_utilization": e("VLLM_GPU_MEMORY_UTILIZATION"),
             "flags": e("VLLM_FLAGS"), "flags_source": e("VLLM_FLAGS_SOURCE") or None,
             "profile": e("VLLM_PROFILE") or None, "command": e("VLLM_CMDLINE") or None,
             "probe_extra": probe_extra,
             "probe_extra_source": e("VLLM_PROBE_EXTRA_SOURCE") or None,
             "ready_s": e("READY_S") or None},
    "probe": probe,
    "grade": grade,
    "endpoint": {"job_id": e("SWARMLAB_ENDPOINT_JOB") or None,
                 "base_url": e("SWARMLAB_ENDPOINT_URL")} if e("SWARMLAB_ENDPOINT_URL") else None,
}
print(json.dumps(doc, indent=1))
PY
  hf_q buckets cp "$JT/job.json" "$SWARMLAB_STAGE/job.json" || log "WARNING: manifest upload failed"
}

# ---- bucket sync ------------------------------------------------------------------------------
sync_runs() {
  [ "$MODE" != grade ] && [ -d "$OUT" ] || return 0
  if command -v flock >/dev/null; then
    flock "$JT/swarmlab-sync.lock" "$HF" buckets sync "$OUT" "$SWARMLAB_RUNS_REMOTE" >>"$JT/joblogs/sync.log" 2>&1
  else
    "$HF" buckets sync "$OUT" "$SWARMLAB_RUNS_REMOTE" >>"$JT/joblogs/sync.log" 2>&1
  fi || log "WARNING: bucket sync failed (tail: $(tail -2 "$JT/joblogs/sync.log" | tr '\n' ' '))"
}

finish() {
  trap - EXIT TERM INT
  for pid in "${!RUNNING[@]}"; do kill -TERM "$pid" 2>/dev/null; done
  for pid in "${!RUNNING[@]}"; do wait "$pid" 2>/dev/null; done
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
  if [ "$STATUS" != "completed" ]; then STATUS="failed"; fi
  [ "$RC" -eq 0 ] && [ "$STATUS" = "failed" ] && RC=1
  log "finishing: status=$STATUS rc=$RC; final sync of $OUT -> ${SWARMLAB_RUNS_REMOTE:-(none)}"
  sync_runs
  if [ -n "$VLLM_PID" ]; then
    grep -c -i preempt "$JT/vllm.log" 2>/dev/null | sed 's/^/vLLM preemption lines: /'
    kill "$VLLM_PID" 2>/dev/null
  fi
  [ -f "$JT/vllm.log" ] && hf_q buckets cp "$JT/vllm.log" "$SWARMLAB_STAGE/vllm.log"
  [ -f "$JT/results.jsonl" ] && hf_q buckets cp "$JT/results.jsonl" "$SWARMLAB_STAGE/results.jsonl"
  hf_q buckets sync "$JT/joblogs" "$SWARMLAB_STAGE/logs"
  write_manifest "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "JOB_EXIT $RC"
  exit "$RC"
}
trap 'RC=$?; finish' EXIT
trap 'log "signal: stopping (job timeout or cancel)"; RC=143; finish' TERM INT
fail() { RC="$1"; shift; log "FATAL: $*"; exit "$RC"; }
[[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || fail 10 "SWARMLAB_PARALLEL must be an integer >= 1, got $PARALLEL"
# only the slot pool needs `wait -n -p`; one run at a time works on older bash (task images)
[ "$PARALLEL" -eq 1 ] || [ $(( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] )) -ge 501 ] \
  || fail 10 "bash >= 5.1 needed for SWARMLAB_PARALLEL > 1 (wait -n -p)"

log "mode=$MODE tag=$SWARMLAB_TAG job=${JOB_ID:-?} commit=${SWARMLAB_GIT_COMMIT:-?} accelerator=${ACCELERATOR:-?}"
if [ "$MODE" = vllm ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || log "no GPU visible"
fi

# ---- api / grade: uv and a managed Python 3.12 (the image may have neither) -------------------
if [ "$MODE" != vllm ]; then
  if ! command -v "$UV" >/dev/null 2>&1; then
    log "installing uv into $ROOT/uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/uv" UV_NO_MODIFY_PATH=1 sh \
      >>"$JT/joblogs/install.log" 2>&1 || fail 10 "uv install failed"
    UV="$ROOT/uv/uv"
  fi
  # kept under ROOT (root-only), not linked into ~/.local/bin, not on PATH
  export UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_PYTHON_BIN_DIR="$ROOT/python/bin"
  log "installing a managed Python 3.12 into $UV_PYTHON_INSTALL_DIR"
  uv python install 3.12 >>"$JT/joblogs/install.log" 2>&1 || fail 10 "uv python install 3.12 failed"
fi

# ---- swarmlab (wheel from the stage dir) + hf CLI in a venv ----------------------------------
WHEEL_FILE=$(ls "$STAGE_DIR"/swarmlab-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL_FILE" ] || fail 10 "no swarmlab wheel in $STAGE_DIR"
WHEEL="$WHEEL_FILE"
[ -n "${SWARMLAB_EXTRAS:-}" ] && WHEEL="swarmlab[$SWARMLAB_EXTRAS] @ file://$WHEEL_FILE"
log "installing $(basename "$WHEEL_FILE")${SWARMLAB_EXTRAS:+ [$SWARMLAB_EXTRAS]} + huggingface_hub into $VENV"
uv venv -q --python 3.12 "$VENV" || fail 10 "uv venv failed"
uv pip install -q --python "$VENV/bin/python" "$WHEEL" "huggingface_hub>=1.32" || fail 10 "pip install of the wheel failed"
unset UV_PYTHON_INSTALL_DIR UV_PYTHON_BIN_DIR   # (api/grade) not needed past the install
HF="$VENV/bin/hf"
SWARMLAB="$VENV/bin/swarmlab"
if [ "$MODE" != grade ]; then
  "$SWARMLAB" validate "$STAGE_DIR/spec.yaml" >/dev/null || fail 10 "spec.yaml does not validate"
fi
write_manifest

# ---- vLLM (vllm mode only) --------------------------------------------------------------------
if [ "$MODE" = vllm ]; then
STATUS=serving
log "uv pip install --system vllm==$VLLM_VERSION"
uv pip install --system -q "vllm==$VLLM_VERSION" || fail 11 "vLLM install failed"
VLLM_BIN=$(command -v vllm) || fail 11 "vllm executable not on PATH after install"
# the argv the plan recorded (plan.json `vllm_serve_argv`), checked against the env, written
# NUL-delimited for mapfile; prints its shlex.join (== plan.json `vllm_serve_command`) for the log
VLLM_CMDLINE=$(PLAN_FILE="$STAGE_DIR/plan.json" ARGV_FILE="$JT/vllm_argv.bin" \
  VLLM_PORT="$VLLM_PORT" python3 - <<'PY'
import json, os, shlex, sys
e = os.environ
def die(msg):
    sys.exit("SERVE ARGV ERROR: " + msg)
try:
    argv = json.load(open(e["PLAN_FILE"]))["vllm_serve_argv"]
except (OSError, ValueError, KeyError, TypeError) as x:
    die("no vllm_serve_argv in %s (%s: %s)" % (e["PLAN_FILE"], type(x).__name__, x))
if not (isinstance(argv, list) and len(argv) >= 3 and argv[:2] == ["vllm", "serve"]
        and all(isinstance(a, str) and "\0" not in a for a in argv)):
    die("vllm_serve_argv must be a list of strings starting with vllm serve MODEL: %r" % (argv,))
if argv[2] != e.get("VLLM_MODEL"):
    die("plan.json serves %r but VLLM_MODEL is %r" % (argv[2], e.get("VLLM_MODEL")))
port = argv[argv.index("--port") + 1] if "--port" in argv[:-1] else None
if port != e["VLLM_PORT"]:
    die("plan.json serves on port %r but VLLM_PORT is %r" % (port, e["VLLM_PORT"]))
try:
    flags = shlex.split(e.get("VLLM_FLAGS", ""))
except ValueError as x:
    die("cannot parse VLLM_FLAGS %r: %s" % (e.get("VLLM_FLAGS"), x))
if flags and argv[len(argv) - len(flags):] != flags:
    die("VLLM_FLAGS %r is not the tail of plan.json's vllm_serve_argv" % e.get("VLLM_FLAGS"))
with open(e["ARGV_FILE"], "wb") as f:
    f.write(b"".join(a.encode("utf-8", "surrogateescape") + b"\0" for a in argv[1:]))
print(shlex.join(argv))
PY
) || fail 12 "no usable vllm serve argv in $STAGE_DIR/plan.json (see SERVE ARGV ERROR above)"
mapfile -d '' -t SERVE_ARGS <"$JT/vllm_argv.bin"
log "starting (profile=${VLLM_PROFILE:-none}, flags from ${VLLM_FLAGS_SOURCE:-?}): $VLLM_CMDLINE"
"$VLLM_BIN" "${SERVE_ARGS[@]}" >"$JT/vllm.log" 2>&1 &
VLLM_PID=$!
WAIT_T0=$(date +%s)
while :; do
  if curl -sf "$BASE/health" >/dev/null && curl -sf "$BASE/v1/models" | grep -q "\"$VLLM_MODEL\""; then
    break
  fi
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -40 "$JT/vllm.log"; fail 12 "vLLM exited before it was ready"; }
  if [ $(( $(date +%s) - WAIT_T0 )) -ge "$VLLM_WAIT_S" ]; then
    tail -40 "$JT/vllm.log"; fail 12 "vLLM not ready after ${VLLM_WAIT_S}s (/health and /v1/models)"
  fi
  sleep 5
done
READY_S=$(( $(date +%s) - WAIT_T0 ))
log "vLLM ready in ${READY_S}s: $(grep -E 'KV cache|Maximum concurrency' "$JT/vllm.log" | tail -2 | tr '\n' ' ')"

# one real tool-calling completion: proves the parsers before any run starts. The request extra
# (VLLM_PROBE_EXTRA: --probe-extra, else the serve profile's, else the default) carries
# model-specific fields such as chat_template_kwargs.enable_thinking=false; vLLM drops template
# kwargs a template does not use.
VLLM_PORT="$VLLM_PORT" VLLM_MODEL="$VLLM_MODEL" PROBE_FILE="$JT/probe.json" python3 - <<'PY' \
  || fail 13 "endpoint probe failed (see PROBE ERROR above)"
import json, os, time, urllib.error, urllib.request
try:
    extra = json.loads(os.environ["VLLM_PROBE_EXTRA"])
except ValueError as e:
    extra = "not JSON: %s" % e
if not isinstance(extra, dict):
    print("PROBE ERROR: VLLM_PROBE_EXTRA must be a JSON object, got %r" % os.environ["VLLM_PROBE_EXTRA"][:200])
    raise SystemExit(1)
body = {"model": os.environ["VLLM_MODEL"], "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Guess candidate B with the tool."}],
        "tools": [{"type": "function", "function": {"name": "guess", "description": "Guess the flag.",
                   "parameters": {"type": "object", "properties": {"candidate": {"type": "string"}},
                                  "required": ["candidate"]}}}],
        "tool_choice": "auto", **extra}
out = {"outcome": "error", "extra": extra, "extra_source": os.environ.get("VLLM_PROBE_EXTRA_SOURCE") or None}
def save():
    with open(os.environ["PROBE_FILE"], "w") as f:
        json.dump(out, f)
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % os.environ.get("VLLM_PORT", "8000"),
                             data=json.dumps(body).encode(), headers={"content-type": "application/json"})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=300))
except urllib.error.HTTPError as e:
    out.update(error="HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:500]))
    save()
    print("PROBE ERROR: %s" % out["error"])
    raise SystemExit(1)
except Exception as e:
    out.update(error="%s: %s" % (type(e).__name__, e))
    save()
    print("PROBE ERROR: %s" % out["error"])
    raise SystemExit(1)
ch = r["choices"][0]
calls = ch["message"].get("tool_calls") or []
out.update(outcome="ok" if calls else "no_tool_call", finish_reason=ch.get("finish_reason"),
           tool_calls=len(calls), seconds=round(time.time() - t0, 1), usage=r.get("usage"),
           content=(ch["message"].get("content") or "")[:300])
save()
print("PROBE finish=%s tool_calls=%s usage=%s %.1fs" % (ch.get("finish_reason"), json.dumps(calls)[:200],
      r.get("usage"), time.time() - t0))
if not calls:
    print("SWARMLAB_PROBE_WARNING: no parsed tool call (check --tool-call-parser / --vllm-flags); "
          "runs will continue. content: %r" % out["content"][:200])
PY
fi  # vllm mode

# ---- grade (grade mode only) ------------------------------------------------------------------
if [ "$MODE" = grade ]; then
  STATUS=grading
  write_manifest
  [ -f "$STAGE_DIR/submission.tar.gz" ] || fail 14 "no submission.tar.gz in $STAGE_DIR"
  log "grading $SWARMLAB_GRADE_TASK (tests at $SWARMLAB_GRADE_TESTS)"
  # in the background + `wait`, so a SIGTERM (job timeout) reaches the trap at once
  "$VENV/bin/python" -m swarmlab.jobs.grade --task "$SWARMLAB_GRADE_TASK" \
    --submission "$STAGE_DIR/submission.tar.gz" --tests "$SWARMLAB_GRADE_TESTS" \
    --image "${SWARMLAB_IMAGE:-}" --out "$JT/grade.json" --sweep-on-timeout \
    > >(tee "$JT/joblogs/grade.log") 2>&1 &
  RUNNING[$!]="grade"   # finish() stops it on a SIGTERM
  wait "$!"
  rc=$?
  RUNNING=()
  [ -f "$JT/grade.json" ] || fail 14 "grading failed (exit $rc, no grade.json; see logs/grade.log)"
  hf_q buckets cp "$JT/grade.json" "$SWARMLAB_STAGE/grade.json" || fail 14 "upload of grade.json failed"
  log "grade.json uploaded to $SWARMLAB_STAGE/grade.json"
  STATUS=completed; RC=0
  exit 0
fi

# ---- runs -------------------------------------------------------------------------------------
STATUS=running
write_manifest
( while sleep "$SYNC_EVERY"; do sync_runs; done ) &
SYNC_PID=$!
cd "$WORK" || fail 10 "no work dir"
# a run ends: its summary line, DONE/FAILED, a sync (the shell is single-threaded, so these never
# interleave). `wait -n -p` (bash >= 5.1) returns at a SIGTERM too, so the trap runs at once.
reap_one() {
  local pid rc item
  if [ "${#RUNNING[@]}" -eq 1 ]; then   # one in flight: a plain wait (any bash)
    pid="${!RUNNING[*]}"
    wait "$pid"
  else
    wait -n -p pid "${!RUNNING[@]}"
  fi
  rc=$?
  if [ -z "$pid" ]; then   # no child left to wait for (should not happen): count them failed
    log "WARNING: lost track of runs ${RUNNING[*]} (wait -n exit $rc)"
    FAILED_RUNS="$FAILED_RUNS ${RUNNING[*]}"; RUNNING=(); return
  fi
  item="${RUNNING[$pid]}"
  unset "RUNNING[$pid]"
  arm="${item%:*}"; seed="${item##*:}"
  cat "$JT/joblogs/run-$arm-s$seed.json" >>"$JT/results.jsonl"
  if [ "$rc" -eq 0 ]; then DONE_RUNS="$DONE_RUNS $item"; else FAILED_RUNS="$FAILED_RUNS $item"; log "run $item failed (exit $rc)"; fi
  sync_runs
  hf_q buckets cp "$JT/results.jsonl" "$SWARMLAB_STAGE/results.jsonl"
}
[ "$PARALLEL" -gt 1 ] && log "running up to $PARALLEL runs at once"
for item in $SWARMLAB_RUNS; do
  [ "${#RUNNING[@]}" -ge "$PARALLEL" ] && reap_one
  arm="${item%:*}"; seed="${item##*:}"
  log "run arm=$arm seed=$seed"
  # in the background + `wait`, so a SIGTERM (job timeout) reaches the trap at once
  "$SWARMLAB" run "$STAGE_DIR/spec.yaml" --arm "$arm" --seed "$seed" --yes --out "$OUT" --json \
    >"$JT/joblogs/run-$arm-s$seed.json" 2> >(tee "$JT/joblogs/run-$arm-s$seed.err" >&2) &
  RUNNING[$!]="$item"
done
while [ "${#RUNNING[@]}" -gt 0 ]; do reap_one; done
if [ -z "$FAILED_RUNS" ]; then STATUS=completed; RC=0; else STATUS=failed; RC=1; fi
exit "$RC"

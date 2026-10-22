#!/usr/bin/env bash
# swarmlab serving job (swarmlab/jobs/serve.py). Runs INSIDE the job started by `swarmlab job serve
# --launch`: the job entry fetched the stage dir (serve.sh, plan.json) into $STAGE_DIR, then runs
# `bash serve.sh`. The job IS the model server: set up the engine, start it, wait until it is
# ready, log `SWARMLAB_SERVE_READY url=<local url>`, then wait on the server until the job's
# timeout (or `swarmlab job stop`) ends it. Nothing of swarmlab is installed here.
#
# The setup steps and the serve command are argv lists in $STAGE_DIR/plan.json (`setup_argvs`,
# `serve_argv`: what `swarmlab job serve` printed), executed as is (NUL-delimited, so a token with
# spaces stays one argument). Settings from the environment (`-e`, set by serve.py):
#   SWARMLAB_TAG SWARMLAB_STAGE (hf://buckets/.../jobs/<tag>: serve.json and server.log land here)
#   SWARMLAB_SERVE_ENGINE SWARMLAB_SERVE_MODEL SWARMLAB_SERVE_NAME (the served model name)
#   SWARMLAB_SERVE_PORT  SWARMLAB_SERVE_READY_PATHS (space-separated, polled in order on 127.0.0.1)
#   SWARMLAB_SERVE_WAIT_S (default 1800: the server must be ready this long after it starts)
#   SWARMLAB_SERVE_PROBE_EXTRA (JSON object merged into the one tool-calling probe request)
#   SWARMLAB_UV (the uv the job entry installed or found; used only for `hf buckets cp`)
#   plus the profile's setup env (e.g. PIP_BREAK_SYSTEM_PACKAGES=1 for the sglang kit).
#
# Exit codes (bootstrap.sh's table): 10 plan.json missing or unusable; 11 a setup step failed;
# 12 the server exited before it was ready, or was not ready in time; 13 the probe request failed
# (HTTP error); 15 the server exited after it was ready; 143 SIGTERM (the job's timeout or a
# cancel: the normal end of a serving job). A probe answer without a parsed tool call only warns
# (SWARMLAB_PROBE_WARNING).
set -uo pipefail

T0=$(date +%s)
log() { echo "[swarmlab-serve +$(( $(date +%s) - T0 ))s] $*"; }
: "${SWARMLAB_TAG:?}" "${SWARMLAB_STAGE:?}" "${SWARMLAB_SERVE_PORT:?}" "${SWARMLAB_SERVE_NAME:?}"
ROOT="${SWARMLAB_JOB_ROOT:-/opt/swarmlab}"
STAGE_DIR="${SWARMLAB_STAGE_DIR:-$ROOT/stage}"
JT="${SWARMLAB_JOB_TMP:-$ROOT/tmp}"
PORT="$SWARMLAB_SERVE_PORT"
BASE="http://127.0.0.1:$PORT"
WAIT_S="${SWARMLAB_SERVE_WAIT_S:-1800}"
READY_PATHS="${SWARMLAB_SERVE_READY_PATHS:-/health /v1/models}"
UV="${SWARMLAB_UV:-uv}"
[ -n "${SWARMLAB_UV:-}" ] && export PATH="$(dirname "$SWARMLAB_UV"):$PATH"
export PYTHONUNBUFFERED=1
mkdir -p "$JT"
STATUS=starting RC=0 SERVER_PID="" READY_S="" STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

hf_cp() {   # a throwaway hf CLI (the image's huggingface_hub is left alone)
  "$UV" tool run --from 'huggingface_hub>=1.32' hf buckets cp "$1" "$SWARMLAB_STAGE/$2" \
    >>"$JT/hf.log" 2>&1 || log "WARNING: upload of $2 failed"
}

manifest() {
  ENDED_AT="${1:-}" STATUS="$STATUS" RC="$RC" READY_S="$READY_S" STARTED_AT="$STARTED_AT" \
    PROBE_FILE="$JT/probe.json" python3 - <<'PY' >"$JT/serve.json"
import json, os
e = os.environ.get
try:
    probe = json.load(open(e("PROBE_FILE")))
except (OSError, ValueError):
    probe = None
print(json.dumps({
    "mode": "serve", "tag": e("SWARMLAB_TAG"), "job_id": e("JOB_ID"), "flavor": e("SWARMLAB_FLAVOR"),
    "accelerator": e("ACCELERATOR"), "image": e("SWARMLAB_IMAGE"), "commit": e("SWARMLAB_GIT_COMMIT"),
    "engine": e("SWARMLAB_SERVE_ENGINE"), "model": e("SWARMLAB_SERVE_MODEL"),
    "served_model_name": e("SWARMLAB_SERVE_NAME"), "port": int(e("SWARMLAB_SERVE_PORT")),
    "started_at": e("STARTED_AT"), "ended_at": e("ENDED_AT") or None, "status": e("STATUS"),
    "exit_code": int(e("RC") or 0), "ready_s": int(e("READY_S")) if e("READY_S") else None,
    "probe": probe}, indent=1))
PY
  hf_cp "$JT/serve.json" serve.json
}

finish() {
  trap - EXIT TERM INT
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; fi
  [ "$RC" -eq 143 ] && STATUS=stopped
  [ "$STATUS" = stopped ] || STATUS=failed
  [ -f "$JT/setup.log" ] && hf_cp "$JT/setup.log" logs/setup.log
  [ -f "$JT/server.log" ] && hf_cp "$JT/server.log" logs/server.log
  manifest "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "JOB_EXIT $RC"
  exit "$RC"
}
trap 'RC=$?; finish' EXIT
trap 'log "signal: stopping (job timeout or cancel)"; RC=143; finish' TERM INT
fail() { RC="$1"; shift; log "FATAL: $*"; exit "$RC"; }

log "serve tag=$SWARMLAB_TAG job=${JOB_ID:-?} engine=${SWARMLAB_SERVE_ENGINE:-?} model=${SWARMLAB_SERVE_MODEL:-?} accelerator=${ACCELERATOR:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || log "no GPU visible"

# the planned argv lists, one NUL-delimited file per step; prints each step's shlex.join
PLAN_FILE="$STAGE_DIR/plan.json" OUT_DIR="$JT" python3 - <<'PY' || fail 10 "no usable setup_argvs/serve_argv in $STAGE_DIR/plan.json"
import json, os, shlex, sys
e = os.environ
try:
    plan = json.load(open(e["PLAN_FILE"]))
    steps, serve = plan["setup_argvs"], plan["serve_argv"]
except (OSError, ValueError, KeyError, TypeError) as x:
    sys.exit("SERVE PLAN ERROR: %s: %s" % (type(x).__name__, x))
def ok(argv):
    return isinstance(argv, list) and argv and all(isinstance(a, str) and "\0" not in a for a in argv)
if not (isinstance(steps, list) and all(ok(a) for a in steps) and ok(serve)):
    sys.exit("SERVE PLAN ERROR: setup_argvs must be a list of argv lists and serve_argv an argv list")
def write(name, argv):
    with open(os.path.join(e["OUT_DIR"], name), "wb") as f:
        f.write(b"".join(a.encode("utf-8", "surrogateescape") + b"\0" for a in argv))
for i, argv in enumerate(steps):
    write("setup-%02d.bin" % i, argv)
write("serve.bin", serve)
print("plan: %d setup step(s); serve: %s" % (len(steps), shlex.join(serve)))
PY

STATUS=setup
manifest
for f in "$JT"/setup-*.bin; do
  [ -e "$f" ] || continue
  mapfile -d '' -t STEP <"$f"
  log "setup: ${STEP[*]}"
  "${STEP[@]}" >>"$JT/setup.log" 2>&1 || { tail -40 "$JT/setup.log"; fail 11 "setup step failed: ${STEP[*]}"; }
done

mapfile -d '' -t SERVE <"$JT/serve.bin"
log "starting: ${SERVE[*]}"
"${SERVE[@]}" >"$JT/server.log" 2>&1 &
SERVER_PID=$!
STATUS=starting
WAIT_T0=$(date +%s)
while :; do
  ready=1
  for path in $READY_PATHS; do
    curl -sf "$BASE$path" -o "$JT/ready.out" || { ready=0; break; }
    if [ "$path" = /v1/models ] && ! grep -q "\"$SWARMLAB_SERVE_NAME\"" "$JT/ready.out"; then ready=0; break; fi
  done
  [ "$ready" -eq 1 ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -60 "$JT/server.log"; SERVER_PID=""; fail 12 "the server exited before it was ready"; }
  if [ $(( $(date +%s) - WAIT_T0 )) -ge "$WAIT_S" ]; then
    tail -40 "$JT/server.log"; fail 12 "the server was not ready after ${WAIT_S}s ($READY_PATHS)"
  fi
  sleep 10
done
READY_S=$(( $(date +%s) - WAIT_T0 ))
log "SWARMLAB_SERVE_READY url=$BASE served_model_name=$SWARMLAB_SERVE_NAME ready_s=$READY_S"

# one tool-calling request: proves the parsers (a missing tool call only warns)
PROBE_FILE="$JT/probe.json" python3 - <<'PY' || fail 13 "the probe request failed (see PROBE ERROR above)"
import json, os, time, urllib.error, urllib.request
e = os.environ
extra = json.loads(e.get("SWARMLAB_SERVE_PROBE_EXTRA") or "{}")
body = {"model": e["SWARMLAB_SERVE_NAME"], "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Guess candidate B with the tool."}],
        "tools": [{"type": "function", "function": {"name": "guess", "description": "Guess the flag.",
                   "parameters": {"type": "object", "properties": {"candidate": {"type": "string"}},
                                  "required": ["candidate"]}}}],
        "tool_choice": "auto", **extra}
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % e["SWARMLAB_SERVE_PORT"],
                             data=json.dumps(body).encode(), headers={"content-type": "application/json"})
t0 = time.time()
out = {"outcome": "error"}
try:
    r = json.load(urllib.request.urlopen(req, timeout=900))
except Exception as x:
    detail = x.read().decode("utf-8", "replace")[:500] if isinstance(x, urllib.error.HTTPError) else ""
    out["error"] = "%s: %s %s" % (type(x).__name__, x, detail)
    json.dump(out, open(e["PROBE_FILE"], "w"))
    print("PROBE ERROR: %s" % out["error"])
    raise SystemExit(1)
msg = r["choices"][0]["message"]
calls = msg.get("tool_calls") or []
out = {"outcome": "ok" if calls else "no_tool_call", "finish_reason": r["choices"][0].get("finish_reason"),
       "tool_calls": len(calls), "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
       "seconds": round(time.time() - t0, 1), "usage": r.get("usage")}
json.dump(out, open(e["PROBE_FILE"], "w"))
print("PROBE %s" % json.dumps(out))
if not calls:
    print("SWARMLAB_PROBE_WARNING: no parsed tool call (finish %s)" % out["finish_reason"])
PY

STATUS=serving
manifest
log "serving until the job's timeout or \`swarmlab job stop\` (server pid $SERVER_PID)"
wait "$SERVER_PID"
rc=$?
SERVER_PID=""
tail -40 "$JT/server.log"
fail 15 "the server exited (exit $rc) after it was ready"

"""Synthetic cache regression for a dedicated localhost test instance only.

Default: the original six checks. --extended adds long restore and cancellation.
Long coverage assumes the test instance's 32768 budget + 16384 gen reserve;
timings must prove >49152 input tokens AND >49152 restored cached tokens.
The API does not expose the sparse attention mask; retain server trace alongside
these results to confirm the internal sparse path. No process is started here.

--eviction-only runs only eviction cases at --port, after the controller has
serially restarted that same instance with --kvmem-pool-max 1. No second model
or port is needed. This tests count eviction, not byte-budget eviction.
--expect-gpu-reuse [on|off] additionally checks the mode and actual per-run
resident_stats deltas; a configuration flag or prompt-cache hit is not evidence.
Cold long requests can take 150s; the default per-request timeout is 360s.
"""

import argparse
import copy
import json
from pathlib import Path
import re
import socket
import time
import urllib.request


def request(port, path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data, {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    return value, time.monotonic() - started


def fixture():
    system = "Synthetic cache test. Answer with one short sentence.\n" + "\n".join(
        f"Record {i:04d}: project Cedar owns item {i:04d}; the agreed color is green and its status is pending."
        for i in range(180)
    )
    user = "Use the tool observations to report the final status of project Cedar.\n" + "\n".join(
        f"Question context {i:03d}: the pending item must be checked against the latest observation."
        for i in range(80)
    )
    base = {
        "model": "default", "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Branch A historical request. Keep the project record."},
            {"role": "assistant", "content": "Historical record:\n" + "\n".join(
                f"Archived item {i:03d} belongs to Cedar; this is stable history, not a current status observation."
                for i in range(70))},
            {"role": "user", "content": "Retain that history and use future tool observations for the final status."},
            {"role": "assistant", "content": "The history is retained. I will use the latest tool observation."},
            {"role": "user", "content": user}],
        "tools": [{"type": "function", "function": {"name": "lookup_status", "description": "Get item status",
                   "parameters": {"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]}}}],
        "temperature": 0, "seed": 7, "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    continuation = copy.deepcopy(base)
    continuation["messages"] += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_cedar", "type": "function",
         "function": {"name": "lookup_status", "arguments": '{"project":"Cedar"}'}}]},
        {"role": "tool", "tool_call_id": "call_cedar", "content": "Project Cedar: approved. Owner: Lin. Color: green."},
    ]
    other = {"model": "default", "messages": [
        {"role": "system", "content": "You are a separate test conversation. Reply exactly BLUE."},
        {"role": "user", "content": "What is the marker?"}], "temperature": 0, "seed": 7, "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False}}
    return base, continuation, other


LONG_THRESHOLD = 49152
FACTS = ("approved", "Lin", "green")


def fact_fixture(label, records, facts=FACTS):
    """Long system boundary allows a parked resume beyond the sparse threshold."""
    history = "\n".join(
        f"Archive row {i:05d}: parcel belongs to depot west; this historical inventory is unrelated to the current project."
        for i in range(records)
    )
    return {
        "model": "default", "messages": [
            {"role": "system", "content": (
                f"Independent project {label}. Archive rows are background only.\n"
                "Use the authoritative current facts below. Answer with exactly three values in this order: "
                "status owner color. Separate values with spaces; no labels or explanation.\n"
                + history + f"\nAuthoritative current facts for {label}: "
                f"status={facts[0]}; owner={facts[1]}; color={facts[2]}."
            )},
            {"role": "user", "content": f"Report the current status, owner and color of {label}."},
        ],
        "temperature": 0, "seed": 7, "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def token_counts(value):
    timings = value.get("timings", {})
    prompt, cached = timings.get("prompt_n"), timings.get("cache_n")
    if any(type(n) is not int or n < 0 for n in (prompt, cached)):
        raise ValueError("Missing or invalid timings.prompt_n/cache_n")
    return {"prompt_n": prompt, "cache_n": cached, "total_input_tokens": prompt + cached}


def fact_content(value, facts):
    message = value["choices"][0]["message"]
    content = message.get("content")
    words = re.findall(r"[a-z]+", content.casefold()) if isinstance(content, str) else []
    return not message.get("tool_calls") and words == [word.casefold() for word in facts]


def slots(port):
    value, _ = request(port, "/slots", timeout=10)
    if (not isinstance(value, list) or len(value) != 1
            or type(value[0].get("is_processing")) is not bool):
        raise ValueError("Expected exactly one test inference slot")
    return value[0]


def wait_idle(port, timeout):
    deadline = time.monotonic() + timeout
    next_notice = 0
    while time.monotonic() < deadline:
        state = slots(port)
        if not state["is_processing"]:
            return state
        if time.monotonic() >= next_notice:
            print("Waiting for cancelled request to release its slot", flush=True)
            next_notice = time.monotonic() + 5
        time.sleep(0.5)
    raise TimeoutError("Cancelled request did not release its slot")


def cancel_prefill(port, body, timeout, evidence):
    """Disconnect only after observing a new task actively doing partial prefill.

    Do not wait for a content token: that would cancel after the long prefill.
    Raw HTTP keeps the socket under our control, without a buffered SSE reader.
    """
    before = slots(port)
    if before["is_processing"]:
        raise RuntimeError("Cancellation test requires an idle dedicated instance")
    payload = json.dumps(dict(body, stream=True, cache_reset=True, pool_reset=True)).encode()
    headers = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    observed = False
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as peer:
        try:
            peer.sendall(headers + payload)
            deadline, next_notice = time.monotonic() + timeout, 0
            while time.monotonic() < deadline:
                state = slots(port)
                if state.get("id_task") != before.get("id_task") and state["is_processing"]:
                    processed = state.get("n_prompt_tokens_processed", 0)
                    total = state.get("n_prompt_tokens", 0)
                    if total > LONG_THRESHOLD and 8192 <= processed < total:
                        evidence["at_disconnect"] = state
                        observed = True
                        break
                if time.monotonic() >= next_notice:
                    print("Waiting to cancel a verified >49152-token partial prefill", flush=True)
                    next_notice = time.monotonic() + 5
                time.sleep(0.25)
        finally:
            # SHUT_RDWR propagates through any port forward before the close.
            try:
                peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # A peer that already closed cannot be cancelled twice.
    evidence["disconnect_wall_s"] = time.monotonic() - started
    evidence["partial_prefill_observed"] = observed
    evidence["idle_after_disconnect"] = wait_idle(port, timeout)
    evidence["released_wall_s"] = time.monotonic() - started
    return observed


def cancel_live_prefill(port, body, timeout, evidence, after_s=10.0):
    """Disconnect a prefill that keeps the live conversation, like a client timeout."""
    before = slots(port)
    if before["is_processing"]:
        raise RuntimeError("Cancellation test requires an idle dedicated instance")
    payload = json.dumps(dict(body, stream=True)).encode()
    headers = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    observed, started = False, time.monotonic()
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as peer:
        try:
            peer.sendall(headers + payload)
            first = None
            while time.monotonic() - started < timeout:
                state = slots(port)
                if state.get("id_task") != before.get("id_task") and state["is_processing"]:
                    first = first or time.monotonic()
                    if time.monotonic() - first >= after_s:
                        evidence["at_disconnect"] = state
                        observed = True
                        break
                elif first is not None:
                    break  # Finished before the cancel point.
                time.sleep(0.25)
        finally:
            try:
                peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    evidence["disconnect_wall_s"] = time.monotonic() - started
    evidence["idle_after_disconnect"] = wait_idle(port, timeout)
    return observed


def extended_cases(args, results, run, check):
    extended = results["extended"] = {
        "expected_facts": dict(zip(("status", "owner", "color"), FACTS)),
        "expected_content": "approved Lin green",
        "length_threshold": LONG_THRESHOLD,
        "sparse_configuration_assumption": {"kvmem_budget": 32768, "gen_reserve": 16384},
        "comparison": "normalized factual content; no cross-batch bitwise/COW verdict",
    }

    def facts(name, value, expected=FACTS):
        check(name, fact_content(value, expected), expected_content=" ".join(expected),
              actual_content=value["choices"][0]["message"].get("content"))

    long_body = fact_fixture("CedarLong", args.long_records)
    cold = run("extended_long_cold", dict(long_body, cache_reset=True, pool_reset=True))
    cold_counts = token_counts(cold)
    extended["cold_counts"] = cold_counts
    long_enough = cold_counts["total_input_tokens"] > LONG_THRESHOLD
    check("extended long input exceeds 49152 measured tokens", long_enough, **cold_counts)
    facts("extended cold content has expected facts", cold)
    if not long_enough:
        raise ValueError("Long fixture is too short: increase --long-records; sparse coverage not exercised")

    _, _, interloper = fixture()
    run("extended_long_interloper", interloper)
    parked, _ = request(args.port, "/kvmem/pool", timeout=args.request_timeout)
    extended["pool_parked"] = parked
    long_entries = [e for e in parked.get("entries", []) if e.get("rows", 0) > LONG_THRESHOLD]
    check("extended long entry survives interloper", bool(long_entries))
    restored = run("extended_long_restored", long_body)
    restored_counts = token_counts(restored)
    extended["restored_counts"] = restored_counts
    check("extended restored input has same measured long length",
          restored_counts["total_input_tokens"] == cold_counts["total_input_tokens"], **restored_counts)
    check("extended parked restore reuses beyond 49152 tokens",
          bool(long_entries) and restored_counts["cache_n"] > LONG_THRESHOLD, **restored_counts)
    facts("extended restored content has expected facts", restored)

    if not args.branch_focus:
        evidence = extended["cancel"] = {}
        observed = cancel_prefill(args.port, long_body, args.request_timeout, evidence)
        check("extended cancellation interrupts a verified partial long prefill", observed)
        check("extended cancellation releases inference slot", not evidence["idle_after_disconnect"]["is_processing"])
        health, _ = request(args.port, "/health", timeout=10)
        check("extended health survives cancellation", health.get("status") == "ok")
        retry = run("extended_cancel_retry", long_body)
        retry_counts = token_counts(retry)
        extended["retry_counts"] = retry_counts
        check("extended retry resumes a completed checkpoint", retry_counts["cache_n"] >= 4096, **retry_counts)
        check("extended retry retains measured long input length",
              retry_counts["total_input_tokens"] == cold_counts["total_input_tokens"], **retry_counts)
        facts("extended retry content has expected facts", retry)

    # Production regression: a request sharing the long live prefix but with a
    # different final user message parks the >49152-row live entry (reason=branch).
    def branch_body(question):
        body = copy.deepcopy(long_body)
        body["messages"][-1]["content"] = question
        return body

    branch = extended["branch"] = {}
    done = run("extended_branch_after_done",
               branch_body("Report the current status, owner and color of CedarLong, in that order."))
    branch["after_done"] = token_counts(done)
    check("extended branch after done reuses the long prefix", branch["after_done"]["cache_n"] > LONG_THRESHOLD,
          **branch["after_done"])
    facts("extended branch after done content has expected facts", done)
    # Diverge inside the history (~60%), as when a client rewrites older
    # context. The long live branch must be parked, then resumable from the pool.
    diverged = copy.deepcopy(long_body)
    lines = diverged["messages"][0]["content"].split("\n")
    cut = int(len(lines) * 0.6)
    lines[cut:] = [x.replace("depot west", "depot east") for x in lines[cut:]]
    diverged["messages"][0]["content"] = "\n".join(lines)
    original = branch_body("CedarLong: report current status, owner and color.")

    mid = run("extended_branch_mid_done", diverged)
    branch["mid_done"] = token_counts(mid)
    back = run("extended_branch_back_after_mid", original)
    branch["back_after_mid"] = token_counts(back)
    check("extended original survives a mid-history branch", branch["back_after_mid"]["cache_n"] > LONG_THRESHOLD,
          **branch["back_after_mid"])
    facts("extended back after mid content has expected facts", back)

    evidence = branch["cancel"] = {}
    rewritten = copy.deepcopy(long_body)
    lines = rewritten["messages"][0]["content"].split(chr(10))
    cut = int(len(lines) * 0.5)
    lines[cut:] = [x.replace("depot west", "depot north") for x in lines[cut:]]
    rewritten["messages"][0]["content"] = chr(10).join(lines)
    branch["cancel_observed"] = cancel_live_prefill(args.port, rewritten, args.request_timeout, evidence)
    check("extended live cancellation interrupted a running prefill", branch["cancel_observed"], **evidence)
    after = run("extended_branch_after_cancel", original)
    branch["after_cancel"] = token_counts(after)
    check("extended original survives a cancelled mid-history branch", branch["after_cancel"]["cache_n"] > LONG_THRESHOLD,
          **branch["after_cancel"])
    facts("extended branch after cancel content has expected facts", after)


def pic_body(system, shared, tail):
    return {"model": "default", "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": shared},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": tail}],
        "temperature": 0, "seed": 7, "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False}}


def pic_cases(args, results, run, check):
    """Exercise the approximate server-side segment path.

    A cold request builds one message-aligned segment; a second request whose
    system message diverges early but repeats the same message body must reuse
    it. Output equality with a non-PIC run is NOT asserted: the path is
    approximate by design.
    """
    port = args.port
    pic = results["pic"] = {}
    shared = "\n".join(
        f"Observation {i:04d}: depot west parcel {i:04d} was scanned and its seal is intact; "
        "this observation is background only and is not a current status."
        for i in range(90)
    )
    tail = "Reply with the single word READY and nothing else."
    cold_body = pic_body("PIC scenario cold. Archive rows are background only.", shared, tail)
    warm_body = pic_body("PIC scenario warm. Archive rows are background only.", shared, tail)
    cold = run("pic_cold", dict(cold_body, cache_reset=True, pool_reset=True), port=port)
    pic["cold"] = {"prompt_n": cold.get("timings", {}).get("prompt_n"),
                   "cache_n": cold.get("timings", {}).get("cache_n"),
                   "prompt_ms": cold.get("timings", {}).get("prompt_ms"),
                   "content": cold["choices"][0]["message"].get("content")}
    warm = run("pic_reuse", warm_body, port=port)
    pic["warm"] = {"prompt_n": warm.get("timings", {}).get("prompt_n"),
                   "cache_n": warm.get("timings", {}).get("cache_n"),
                   "prompt_ms": warm.get("timings", {}).get("prompt_ms"),
                   "content": warm["choices"][0]["message"].get("content")}
    check("pic cold request answered", isinstance(pic["cold"]["content"], str) and pic["cold"]["content"].strip() != "")
    check("pic reuse request answered", isinstance(pic["warm"]["content"], str) and pic["warm"]["content"].strip() != "")
    # timings.prompt_n is the total input, not the evaluated subset, so the
    # reduction shows up in prefill time. The controller separately asserts that
    # the server log recorded an actual PIC hit.
    cold_ms = pic["cold"]["prompt_ms"] or 0.0
    warm_ms = pic["warm"]["prompt_ms"] or 0.0
    check("pic reuse lowers prefill time", 0 < warm_ms < cold_ms,
          cold_prompt_ms=cold_ms, warm_prompt_ms=warm_ms)

def eviction_cases(args, results, run, check):
    port = args.port
    small = results["eviction"] = {"port": port, "eviction_kind": "entry_count"}
    if slots(port)["is_processing"]:
        raise RuntimeError("Smallpool instance is busy")
    bodies = [fact_fixture(label, 320, facts=expected) for label, expected in (
        ("EvictA", FACTS), ("EvictB", ("shipped", "Mei", "blue")), ("EvictC", ("held", "Tao", "gold")),
    )]
    seed = run("extended_smallpool_A", dict(bodies[0], cache_reset=True, pool_reset=True), port=port)
    check("extended smallpool seed has expected facts", fact_content(seed, FACTS), expected_content=" ".join(FACTS))
    # A fresh server publishes {} until its first request finishes.
    config, _ = request(port, "/kvmem/pool", timeout=args.request_timeout)
    small["initial_pool"] = config
    configured = config.get("enabled") is True and config.get("max_entries") == 1
    check("extended smallpool instance has max_entries 1", configured)
    if not configured:
        raise ValueError("Eviction-only requires this instance to use --kvmem-pool-max 1")
    run("extended_smallpool_B", bodies[1], port=port)
    before, _ = request(port, "/kvmem/pool", timeout=args.request_timeout)
    small["before_eviction"] = before
    entries = before.get("entries", [])
    check("extended smallpool A is parked before pressure", len(entries) == 1 and entries[0].get("id") is not None)
    run("extended_smallpool_C", bodies[2], port=port)
    after, _ = request(port, "/kvmem/pool", timeout=args.request_timeout)
    small["after_eviction"] = after
    remaining = after.get("entries", [])
    evicted = (len(entries) == 1 and len(remaining) == 1
               and remaining[0].get("id") is not None and remaining[0]["id"] != entries[0].get("id"))
    check("extended smallpool pressure evicts parked A", evicted)
    check("extended smallpool obeys entry cap",
          before.get("max_entries") == after.get("max_entries") == 1 and len(entries) <= 1 and len(remaining) <= 1)
    replay = run("extended_smallpool_A_after_eviction", bodies[0], port=port)
    seed_counts, replay_counts = token_counts(seed), token_counts(replay)
    small["replay_counts"] = replay_counts
    check("extended evicted branch replays same input with cache miss",
          evicted and replay_counts["total_input_tokens"] == seed_counts["total_input_tokens"]
          and replay_counts["cache_n"] < 1024, **replay_counts)
    check("extended evicted branch content has expected facts", fact_content(replay, FACTS), expected_content=" ".join(FACTS))


# ---------------------------------------------------------------- PIC quality
# F3c gate. The same fixed request list is graded on an exact-only server
# (--kvmem-pic off) and on a PIC server (--kvmem-pic on). Each PIC case repeats
# the baseline case body but diverges in the system line, so exact-prefix reuse
# cannot cover the shared body and only a PIC segment splice can. The phase
# writes raw grades; the controller compares the two phases so no approximation
# verdict is made by the instance being tested.
QUALITY_TRUTH = {"status": "approved", "owner": "Lin", "color": "green"}
QUALITY_FALSE = {"status": "rejected", "owner": "Nobody", "color": "red"}

QUALITY_PROMPTS = {
    "tool_and_schema": (
        "Call the tool lookup_status exactly once with project=\"Cedar\". Do not answer in prose.", "tool"),
    "facts_negation": (
        "The archive contains a superseded record and an authoritative one. Report the authoritative "
        "current status, owner and color as exactly three space separated values, nothing else.", "facts"),
    "long_memory": (
        "Report the authoritative current status, owner and color recorded in the shared body as exactly "
        "three space separated values, nothing else.", "facts"),
    "position_move": (
        "Report the authoritative current status, owner and color recorded in the shared body as exactly "
        "three space separated values, nothing else.", "facts"),
    "repeat_emphasis": (
        "Report the REQUIRED OUTPUT token from the shared body. Output that token alone.", "delta"),
    "plain_chat": (
        "In one short friendly sentence, say hello and confirm you are ready to help.", "chat"),
}


def quality_doc(category, index, rows=54):
    """Identical for both phases; only the system line differs between them."""
    lines = [f"Archive block {category} case {index}. The rows below are background only."]
    planted = {
        "facts_negation": 4,
        "long_memory": rows - 6,
        "position_move": (index % 3) * (rows // 3) + 2,
        "repeat_emphasis": rows - 3,
    }.get(category, rows // 2)
    for i in range(rows):
        if i == planted:
            if category == "facts_negation":
                lines.append(
                    f"Row {i:03d}: the earlier note that the current status is {QUALITY_FALSE['status']} "
                    f"with owner {QUALITY_FALSE['owner']} is superseded and must NOT be reported.")
                lines.append(
                    f"Row {i:03d}b: the authoritative current record is status={QUALITY_TRUTH['status']}, "
                    f"owner={QUALITY_TRUTH['owner']}, color={QUALITY_TRUTH['color']}.")
            elif category == "repeat_emphasis":
                lines.append(
                    f"Row {i:03d}: REQUIRED OUTPUT token is DELTA-{index:02d}. "
                    f"Report DELTA-{index:02d} and nothing else.")
            else:
                lines.append(
                    f"Row {i:03d}: the authoritative current record for this case is "
                    f"status={QUALITY_TRUTH['status']}, owner={QUALITY_TRUTH['owner']}, "
                    f"color={QUALITY_TRUTH['color']}.")
        else:
            lines.append(
                f"Row {i:03d}: parcel {i:03d} was scanned at depot west and its seal is intact; "
                "this inventory observation is unrelated and is not a current status.")
    return "\n".join(lines)


def quality_body(category, index, phase):
    prompt, _ = QUALITY_PROMPTS[category]
    guard = ("Respond only from the shared body above. Ignore anything that is not the authoritative record. "
             "Follow the output format exactly; do not add labels, prose or explanations. " * 3)
    return {"model": "default", "messages": [
        {"role": "system", "content": f"Quality scenario {category} phase {phase}. Archive rows are background only."},
        {"role": "user", "content": quality_doc(category, index)},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": prompt + "\n" + guard}],
        "temperature": 0, "seed": 7, "max_tokens": 32,
        "tools": [{"type": "function", "function": {"name": "lookup_status", "description": "Get item status",
                   "parameters": {"type": "object", "properties": {"project": {"type": "string"}},
                                  "required": ["project"]}}}],
        "chat_template_kwargs": {"enable_thinking": False}}


def quality_grade(category, index, value):
    message = value["choices"][0]["message"]
    content = message.get("content") or ""
    words = re.findall(r"[a-z]+", content.casefold())
    truth = [value.casefold() for value in QUALITY_TRUTH.values()]
    if category == "tool_and_schema":
        calls = message.get("tool_calls") or []
        ok = len(calls) == 1 and calls[0].get("function", {}).get("name") == "lookup_status"
        args = {}
        if ok:
            try:
                args = json.loads(calls[0]["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            ok = args.get("project") == "Cedar"
        return ok, {"tool_name": (calls[0].get("function", {}).get("name") if calls else None),
                    "tool_args": args, "content": content[:120]}
    if category == "facts_negation":
        bad = [word for word in (QUALITY_FALSE["status"], QUALITY_FALSE["owner"], QUALITY_FALSE["color"])
               if word.casefold() in content.casefold()]
        return words == truth and not bad, {"words": words, "superseded_leak": bad}
    if category in ("long_memory", "position_move"):
        return words == truth, {"words": words}
    if category == "repeat_emphasis":
        return content.strip().casefold() == f"delta-{index:02d}".casefold(), {"content": content[:120]}
    if category == "plain_chat":
        contamination = [token for token in (*truth, QUALITY_FALSE["status"])
                         if token.casefold() in content.casefold()]
        return bool(words) and not contamination, {"content": content[:120], "contamination": contamination}
    raise ValueError("Unknown quality category")


def quality_phase(args, results, run, check):
    """One server phase only. Writes raw per-case grades; the controller decides."""
    port = args.port
    phase = results["quality_phase"] = {
        "phase": args.quality_phase, "pic": args.pic, "categories": list(QUALITY_PROMPTS),
        "cases_per_category": args.quality_cases, "plan_target_per_category": 20,
        "answers": {},
    }
    answers = phase["answers"]
    for category in QUALITY_PROMPTS:
        answers[category] = []
        for index in range(1, args.quality_cases + 1):
            record = {"index": index}
            body = quality_body(category, index, "graded")
            try:
                if args.pic == "on":
                    # The seed carries a different system line and exists only to
                    # build the segment on a clean evaluation. The graded request
                    # is byte-identical to the graded baseline request, so a PIC
                    # hit is still required (no exact checkpoint for that system
                    # line exists) but nothing else differs between the phases.
                    seed = run(f"quality_{category}_{index}_seed",
                               dict(quality_body(category, index, "seed"), cache_reset=True, pool_reset=True),
                               port=port)
                    if index == 1:
                        phase.setdefault("seed_prompt_ms", {})[category] = seed.get("timings", {}).get("prompt_ms")
                    value = run(f"quality_{category}_{index}_pic", body, port=port)
                else:
                    value = run(f"quality_{category}_{index}_exact",
                                dict(body, cache_reset=True, pool_reset=True), port=port)
                ok, detail = quality_grade(category, index, value)
                record.update(correct=ok, grade=detail,
                              prompt_ms=value.get("timings", {}).get("prompt_ms"))
            except Exception as error:
                record.update(correct=False, error=type(error).__name__)
            answers[category].append(record)
    phase["correct_by_category"] = {
        category: sum(1 for r in rows if r.get("correct") is True) for category, rows in answers.items()}
    check(f"quality phase {args.quality_phase} completed every case",
          all(len(rows) == args.quality_cases for rows in answers.values()))


def percentile(values, rank):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((rank / 100.0) * len(ordered) + 0.5)) - 1))
    return round(ordered[index], 3)


def gpu_reuse_snapshot(port, timeout, expected):
    pool, _ = request(port, "/kvmem/pool", timeout=timeout)
    stats = pool.get("resident_stats", {})
    keys = ("hit_blocks", "miss_blocks", "skipped_target_bytes", "skipped_mtp_bytes")
    if pool.get("gpu_reuse") != expected or stats.get("available") is not True:
        raise ValueError("GPU reuse mode mismatch or actual resident stats unavailable")
    if any(type(stats.get(key)) is not int or stats[key] < 0 for key in keys):
        raise ValueError("Incomplete GPU reuse resident counters")
    return {key: stats[key] for key in keys}


def check_gpu_reuse(args, results, check):
    evidence = results["gpu_reuse"]
    after = gpu_reuse_snapshot(args.port, args.request_timeout, args.expect_gpu_reuse)
    evidence["after"] = after
    delta = {key: after[key] - evidence["before"][key] for key in after}
    evidence["delta"] = delta
    check("gpu reuse counters remain monotonic", all(value >= 0 for value in delta.values()))
    if args.expect_gpu_reuse == "on":
        check("gpu reuse records actual hits and skipped target bytes",
              delta["hit_blocks"] > 0 and delta["skipped_target_bytes"] > 0, **delta)
    else:
        check("gpu reuse off records no hits or skipped bytes",
              all(delta[key] == 0 for key in ("hit_blocks", "skipped_target_bytes", "skipped_mtp_bytes")), **delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=28211)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--extended", action="store_true", help="Add long sparse-restore coverage and cancel/retry")
    modes.add_argument("--eviction-only", action="store_true", help="Only eviction cases on this --port; requires --kvmem-pool-max 1")
    parser.add_argument("--expect-gpu-reuse", nargs="?", const="on", choices=("on", "off"),
                        help="Assert actual resident counter deltas, not just configuration")
    parser.add_argument("--branch-focus", action="store_true",
                        help="With --extended: skip the reset cancel/retry and run only branch coverage")
    parser.add_argument("--long-records", type=int, default=2800, help="Long fixture rows; actual length is checked using timings")
    parser.add_argument("--pic", choices=("on", "off"), help="Server was started with --kvmem-pic; run the segment-reuse cases")
    parser.add_argument("--quality-phase", choices=("baseline", "pic"),
                        help="Run one F3c quality phase only; the controller compares phases")
    parser.add_argument("--quality-cases", type=int, default=2, help="Cases per category (plan target is 20)")
    parser.add_argument("--request-timeout", type=float, default=360, help="Seconds per request; allows 150s cold long prefill")
    args = parser.parse_args()
    if args.port == 18200:
        parser.error("Refusing production port")
    if not 1 <= args.port <= 65535 or args.request_timeout <= 0 or args.long_records <= 0:
        parser.error("Port, request timeout and long-records must be positive and valid")
    if args.quality_phase == "pic" and args.pic != "on":
        parser.error("--quality-phase pic requires --pic on")
    if args.quality_phase == "baseline" and args.pic == "on":
        parser.error("--quality-phase baseline requires a PIC-off server")
    if args.quality_phase and args.pic is None:
        parser.error("--quality-phase requires --pic to state the server mode")
    if not 1 <= args.quality_cases <= 20:
        parser.error("--quality-cases must be between 1 and 20")
    base, continuation, other = fixture()
    results = {"requests": {}, "checks": []}
    if args.expect_gpu_reuse is not None:
        results["gpu_reuse"] = {"expected": args.expect_gpu_reuse}

    def run(name, body, port=None):
        port = args.port if port is None else port
        print(f"START {name} port={port} timeout={args.request_timeout}s", flush=True)
        value, wall = request(port, "/v1/chat/completions", body, timeout=args.request_timeout)
        results["requests"][name] = {"response": value, "wall_s": wall, "port": port}
        timings = value.get("timings", {})
        print(json.dumps({"case": name, "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"), "prompt_ms": timings.get("prompt_ms"), "wall_s": wall}), flush=True)
        if args.expect_gpu_reuse is not None and "before" not in results["gpu_reuse"]:
            # Exclude old process-lifetime hits. The first completed seed also
            # makes a fresh server publish its mode/counters for the first time.
            results["gpu_reuse"]["before"] = gpu_reuse_snapshot(port, args.request_timeout, args.expect_gpu_reuse)
        return value

    def message(value):
        m = copy.deepcopy(value["choices"][0]["message"])
        for call in m.get("tool_calls", []):
            call.pop("id", None)
        return m

    def check(name, ok, **details):
        results["checks"].append({"name": name, "passed": bool(ok), **details})
        print(("PASS " if ok else "FAIL ") + name, flush=True)

    completed = False
    try:
        if args.quality_phase:
            # A quality phase is a complete run of its own; the standard
            # synthetic matrix is unrelated and must not consume its window.
            quality_phase(args, results, run, check)
            completed = True
            return 0 if all(x["passed"] for x in results["checks"]) else 1
        if args.eviction_only:
            eviction_cases(args, results, run, check)
            if args.expect_gpu_reuse is not None:
                check_gpu_reuse(args, results, check)
            completed = True
            return 0 if all(x["passed"] for x in results["checks"]) else 1
        reset = dict(base, cache_reset=True, pool_reset=True)
        run("continuous_seed", reset)
        continuous = run("continuous_tool", continuation)
        run("parked_seed", reset)
        run("interloper", other)
        pool, _ = request(args.port, "/kvmem/pool")
        results["pool_parked"] = pool
        check("entry survives interloper", bool(pool.get("entries")))
        parked = run("parked_tool", continuation)
        check("parked tool prefix hit", parked.get("timings", {}).get("cache_n", 0) > 0)
        a = message(continuous)
        b = message(parked)
        check("tool output matches uninterrupted branch", a == b)
        # Same system, different long branch exercises fork/COW; then return to A.
        sibling = copy.deepcopy(base)
        sibling["messages"][1]["content"] = base["messages"][1]["content"].replace("Branch A", "Branch B")
        run("sibling_fork", sibling)
        fork_pool, _ = request(args.port, "/kvmem/pool")
        results["pool_after_fork"] = fork_pool
        check("parent branch retained after fork", any(e.get("query") for e in fork_pool.get("entries", [])))
        after_fork = run("return_after_fork", continuation)
        check("fork preserves parent output", message(after_fork) == b)
        check("fork preserves parent prefix hit", after_fork.get("timings", {}).get("cache_n", 0) > 0)
        final_pool, _ = request(args.port, "/kvmem/pool")
        results["pool_final"] = final_pool
        if args.extended:
            extended_cases(args, results, run, check)
        if args.quality_phase:
            quality_phase(args, results, run, check)
        elif args.pic == "on":
            pic_cases(args, results, run, check)
        if args.expect_gpu_reuse is not None:
            check_gpu_reuse(args, results, check)
        completed = True
    except Exception as error:
        results["error"] = {"type": type(error).__name__, "message": str(error)}
        check("test execution completed", False)
    finally:
        results["passed"] = completed and all(x["passed"] for x in results["checks"])
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

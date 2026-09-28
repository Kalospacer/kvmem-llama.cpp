#pragma once

// Routing policy for the conversation pool.
//
// Freestanding on purpose, like tools/kvmem-conversation-store.h: no llama.h,
// no ServerState, no logging, no globals. The server computes every input --
// the longest common token prefix in particular, which is media-aware and
// lives in kvmem_prompt::common_prefix -- and executes the plan this file
// returns. This file only decides.
//
//   c++ -std=c++17 -I tools -o /tmp/pool-test tests/conversation-pool-test.cpp && /tmp/pool-test
//
// Two production incidents are pinned here as cases, because both were silent:
//
//   1. A request that shares only the early part of a live conversation used to
//      drop the rest when the dropped span held no checkpoint exactly on a
//      message-start token. One large system message and one user message is
//      enough: the dropped span then holds interval rows only. The conversation
//      was destroyed, not parked, and the next request recomputed 31,932 tokens
//      (75 s) instead of resuming. kPoolDeepBranchRows is that fix.
//   2. Parked entries keep a bounded number of checkpoints. Retaining them
//      oldest-first left every survivor in the rewritten tail, so a rewrite of
//      older context could only resume near the start (68,796 rows recomputed,
//      141 s). The spread lives in pool_keep_checkpoints below.
//
// A check that only holds for one conversation shape is not a check. These
// cases run without a model, a GPU, or a server.

#include <algorithm>
#include <cstdint>
#include <vector>

// Which rows are live on this sequence, and which parked entries are worth
// restoring. One row is one cache row (prompt + generated).
struct pool_live {
    int  lcp      = 0;     // rows of this request's prompt matching the live rows
    int  row      = -1;    // largest checkpoint row that can carry the live branch, -1 if none
    int  rows     = 0;     // live rows
    bool valuable = false; // see pool_branch_worth
};

// One parked conversation, described for this request.
struct pool_entry {
    int  lcp        = 0;   // rows of this request's prompt matching this entry
    int  row        = -1;  // largest checkpoint row at or before the resume cap
    int  rows       = 0;   // stored rows
    bool valuable   = false;
};

// The request is a continuation of the live sequence: it extends it and drops a
// bounded tail, so nothing is abandoned. The bound is absolute, not
// proportional: a per-turn rewritten tail is a few thousand rows, while a
// client rewriting older context abandons history even when the dropped span
// happens to hold no message-start checkpoint.
inline constexpr int kPoolDeepBranchRows = 16384;

// Both thresholds come from the CLI (--kvmem-pool-min-rows, --kvmem-pool-min-gain).
struct pool_policy_limits {
    int min_rows = 2048; // below this an entry is not worth parking
    int min_gain = 1024; // rows that must be gained for a restore to be worth it
};

// Largest checkpoint row at or before the resume cap. A checkpoint is how the
// hybrid model carries recurrent state across a prefix, so a match without one
// cannot be resumed and is not a candidate. Mirrors pool_resume_row.
inline bool pool_checkpoint_row(const std::vector<int> & checkpoint_rows, int lcp, int rows, int cap, int & keep) {
    const int bound = std::min({lcp, rows, cap});
    bool found = false;
    keep = -1;
    for (int row : checkpoint_rows) {
        if (row > 0 && row <= bound && row > keep) {
            keep = row;
            found = true;
        }
    }
    return found;
}

// Would dropping rows (lcp, rows] lose something a later request can resume?
// A checkpoint at a message start means a later turn of that branch lands
// there; the rewritten tail of the current turn only holds query/end rows.
// Dropping far more than a rewritten tail abandons history whatever the
// checkpoints are, which is what case 1 above needed.
inline bool pool_branch_worth(const pool_policy_limits & limits,
                              const std::vector<int> & checkpoint_rows,
                              const std::vector<bool> & checkpoint_is_message_start,
                              int rows, int lcp, bool unrelated) {
    if (rows < limits.min_rows || rows - std::max(lcp, 0) < limits.min_gain) {
        return false;
    }
    const bool deep = rows - std::max(lcp, 0) >= kPoolDeepBranchRows;
    for (size_t i = 0; i < checkpoint_rows.size(); ++i) {
        const int row = checkpoint_rows[i];
        if (row <= lcp || row > rows) continue;
        if ((unrelated || deep) && row >= limits.min_rows) return true;
        if (i < checkpoint_is_message_start.size() && checkpoint_is_message_start[i]) return true;
    }
    return false;
}

// Which checkpoints a parked entry keeps. `rows` is ascending and `is_msg[i]`
// says whether rows[i] sits on a message-start token. The oldest entry, the Q
// anchor (or, without one, the newest row) and the first checkpoint of the
// current turn are load-bearing: the first two are how a later turn and a query
// replay resume, and the third is where a rewrite of the last message lands.
// Everything else is thinned by neighbour distance, message starts last, so
// survivors stay spread over the conversation instead of pooling in the
// rewritten tail (case 2 above). Precondition: cap >= 2, which the CLI enforces.
inline std::vector<int> pool_keep_checkpoints(std::vector<int> rows, std::vector<bool> is_msg,
                                              int cap, int query_anchor, int turn_start) {
    is_msg.resize(rows.size(), false);
    int turn_row = -1;
    if (turn_start >= 0) {
        for (int row : rows) {
            if (row >= turn_start) {
                turn_row = row;
                break;
            }
        }
    }
    while ((int) rows.size() > cap) {
        const auto protect = [&](size_t i) {
            return i == 0 || rows[i] == query_anchor || rows[i] == turn_row ||
                (query_anchor < 0 && i + 1 == rows.size());
        };
        size_t victim = rows.size();
        int best_gap = 0;
        for (int pass = 0; pass < 2 && victim == rows.size(); ++pass) {
            for (size_t i = 1; i < rows.size(); ++i) {
                if (protect(i) || (pass == 0 && is_msg[i])) continue;
                const int gap = i + 1 < rows.size() ? rows[i + 1] - rows[i - 1] : rows[i] - rows[i - 1];
                if (victim == rows.size() || gap < best_gap) {
                    best_gap = gap;
                    victim = i;
                }
            }
        }
        if (victim == rows.size()) {
            // Only protected rows are left (small cap): keep the anchor, drop the turn row first.
            size_t i = 1;
            while (i < rows.size() && !(rows[i] == turn_row && rows[i] != query_anchor)) ++i;
            if (i == rows.size()) {
                i = 1;
                while (i < rows.size() && rows[i] == query_anchor) ++i;
            }
            if (i == rows.size()) break;
            victim = i;
        }
        rows.erase(rows.begin() + (std::ptrdiff_t) victim);
        is_msg.erase(is_msg.begin() + (std::ptrdiff_t) victim);
    }
    return rows;
}

enum class pool_action {
    none,       // nothing to do: run against the live rows as they are
    switch_in,  // restore entry plan.restore (parking live first when plan.park_live)
    park_only,  // park the live conversation and keep only the shared prefix live
};

struct pool_plan {
    pool_action action = pool_action::none;
    int  restore = -1;      // index into the caller's entries, never a pointer
    bool park_live = false; // park before restoring, instead of clearing
    int  resume = -1;       // checkpoint row the restore resumes at
    bool fork = false;      // share the entry's prefix rather than consuming it
    const char * reason = "none";
};

// Pure: no side effects, no logging, no globals. Eviction and restore are
// planned here and executed by the caller.
inline pool_plan pool_route(const pool_live & live,
                            const std::vector<pool_entry> & entries,
                            const pool_policy_limits & limits) {
    pool_plan plan;
    // The best parked entry, by resume row. Ties keep the earliest, which is
    // the most recently parked only because the pool is ordered by insertion.
    int best = -1, best_row = -1;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].row > best_row) {
            best_row = entries[i].row;
            best = (int) i;
        }
    }
    if (best >= 0 && best_row >= std::max(live.row, 0) + limits.min_gain) {
        plan.action = pool_action::switch_in;
        plan.restore = best;
        plan.resume = best_row;
        plan.park_live = live.valuable;
        // Fork when the entry continues well past the shared prefix and the
        // live branch must survive: the caller then shares the prefix instead
        // of consuming the entry.
        plan.fork = entries[best].valuable;
        plan.reason = live.valuable ? "switch" : "switch_clear";
        return plan;
    }
    if (live.valuable) {
        plan.action = pool_action::park_only;
        plan.park_live = true;
        plan.resume = live.row;
        plan.reason = live.row < 0 ? "miss" : "branch";
    }
    return plan;
}

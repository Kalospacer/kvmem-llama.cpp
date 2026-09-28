// Host-side tests for the conversation pool routing policy.
//
//   c++ -std=c++17 -I tools -o /tmp/pool-test tests/conversation-pool-test.cpp && /tmp/pool-test
//
// No model, no GPU, no server. Each case pins a rule and, where the rule was
// learned from a production incident, also runs the pre-fix logic to show the
// case actually fails without the fix. A check that cannot fail is not a check.

#include "kvmem-pool-policy.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } \
} while (0)

// ---- pre-fix logic, kept verbatim so the incident cases can fail -----------

// Before kPoolDeepBranchRows: a dropped span was only worth parking when it
// held a message-start checkpoint. A conversation with one large system message
// and one user message has no such row inside the dropped span.
static bool old_branch_worth(const pool_policy_limits & limits,
                             const std::vector<int> & rows,
                             const std::vector<bool> & is_msg,
                             int total, int lcp, bool unrelated) {
    if (total < limits.min_rows || total - std::max(lcp, 0) < limits.min_gain) return false;
    for (size_t i = 0; i < rows.size(); ++i) {
        const int row = rows[i];
        if (row <= lcp || row > total) continue;
        if (unrelated && row >= limits.min_rows) return true;
        if (i < is_msg.size() && is_msg[i]) return true;
    }
    return false;
}

// Before the spread: survivors were picked oldest-first, which left them all in
// the rewritten tail and forced a resume near the start of the conversation.
static std::vector<int> old_keep_checkpoints(std::vector<int> rows, int cap, int query_anchor,
                                             const std::vector<bool> & is_msg) {
    std::sort(rows.begin(), rows.end());
    const auto is_message = [&](int row) {
        const auto it = std::find(rows.begin(), rows.end(), row);
        return it != rows.end() && is_msg[it - rows.begin()];
    };
    while ((int) rows.size() > cap) {
        auto victim = std::find_if(rows.begin() + 1, rows.end(),
                [&](int row) { return row != query_anchor && !is_message(row); });
        if (victim == rows.end()) {
            victim = std::find_if(rows.begin() + 1, rows.end(), [&](int row) { return row != query_anchor; });
        }
        if (victim == rows.end()) break;
        rows.erase(victim);
    }
    return rows;
}

// ---- the shapes the incidents ran on ---------------------------------------

// One large system message, then one user message: the 72,892-token fixture.
// Interval checkpoints every 4096 rows. None of them is a message start.
struct long_fixture {
    std::vector<int> rows;
    std::vector<bool> is_msg;
    int total = 0;
};

static long_fixture make_long_fixture() {
    long_fixture f;
    for (int row = 4096; row <= 69632; row += 4096) {
        f.rows.push_back(row);
        f.is_msg.push_back(false);
    }
    // The commit row and the query start, as the server records them.
    for (int row : {72870, 72891, 72895}) {
        f.rows.push_back(row);
        f.is_msg.push_back(false);
    }
    f.total = 72895;
    return f;
}

// A message-shaped conversation, for the cases that do have message starts.
static long_fixture make_multi_message_fixture() {
    long_fixture f;
    for (int row : {4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768}) {
        f.rows.push_back(row);
        f.is_msg.push_back(false);
    }
    // Message starts of earlier turns, as AstrBot's turns produce them.
    for (int row : {45056, 51200, 56320}) {
        f.rows.push_back(row);
        f.is_msg.push_back(true);
    }
    // The current turn's rewritten tail sits inside the last user message, so
    // its eval_end and commit rows are not on a message-start token.
    for (int row : {60000, 66560}) {
        f.rows.push_back(row);
        f.is_msg.push_back(false);
    }
    f.total = 66560;
    return f;
}

static void test_incident_deep_branch_is_parked() {
    const pool_policy_limits limits;
    const auto f = make_long_fixture();

    // The request diverges at 60% of the history: lcp is far below the live rows.
    const int lcp = 43713;
    const int live_rows = f.total;

    // Pre-fix: nothing inside the dropped span is a message start, so the
    // conversation was dropped instead of parked. This is the 75 s regression.
    CHECK(!old_branch_worth(limits, f.rows, f.is_msg, live_rows, lcp, false));
    // Fixed: the dropped span is far larger than a rewritten tail.
    CHECK(pool_branch_worth(limits, f.rows, f.is_msg, live_rows, lcp, false));
    // The dropped span really is past the deep threshold, so this case is the
    // threshold and not some incidental checkpoint.
    CHECK(live_rows - lcp >= kPoolDeepBranchRows);
}

static void test_rewritten_tail_is_not_parked() {
    const pool_policy_limits limits;
    // A normal continuation: AstrBot rewrites roughly 7K of temporary tail.
    // The dropped span holds interval rows and possibly a query/commit row, and
    // must NOT be treated as an abandoned branch, or every turn would park.
    const auto f = make_multi_message_fixture();
    const int live_rows = f.total;
    const int lcp = live_rows - 7000;
    CHECK(live_rows - lcp < kPoolDeepBranchRows);
    CHECK(!pool_branch_worth(limits, f.rows, f.is_msg, live_rows, lcp, false));

    // A message-start row inside the dropped span is still worth parking, which
    // is the pre-existing rule and must keep working.
    std::vector<int> rows = {1024, 4096, 8192, 60000};
    std::vector<bool> msg = {false, false, false, true};
    CHECK(pool_branch_worth(limits, rows, msg, 65536, 50000, false));
}

static void test_incident_checkpoint_spread() {
    const auto f = make_long_fixture();
    // Five slots is the production default (--kvmem-pool-ckpts).
    const int cap = 5;
    const int anchor = 72891;
    const int turn_start = 72870; // the last user message starts here

    // Pre-fix: the survivors are all in the tail, so a rewrite of older context
    // can only resume near the start. This is the 141 s regression.
    const auto old_kept = old_keep_checkpoints(f.rows, cap, anchor, f.is_msg);
    const int old_best_resume = *std::max_element(old_kept.begin(), old_kept.end());
    const int old_resume_at_60pct = [&] {
        int best = -1;
        for (int row : old_kept) if (row <= 43713 && row > best) best = row;
        return best;
    }();
    // A request diverging at 60% could only resume at 4096 or below.
    CHECK(old_resume_at_60pct <= 4096);

    // Fixed: survivors are spread, and the turn row survives.
    const auto kept = pool_keep_checkpoints(f.rows, f.is_msg, cap, anchor, turn_start);
    CHECK((int) kept.size() == cap);
    CHECK(std::find(kept.begin(), kept.end(), turn_start) != kept.end());
    CHECK(std::find(kept.begin(), kept.end(), anchor) != kept.end());
    CHECK(std::find(kept.begin(), kept.end(), f.rows.front()) != kept.end());
    // A rewrite at 60% now has somewhere to land, far beyond the start.
    int resume_at_60pct = -1;
    for (int row : kept) if (row <= 43713 && row > resume_at_60pct) resume_at_60pct = row;
    CHECK(resume_at_60pct > 4096);
    CHECK(resume_at_60pct >= old_resume_at_60pct);
    CHECK(old_best_resume >= 0); // the pre-fix set is nonempty; it is just useless
}

static void test_keep_is_bounded_and_ordered() {
    const auto f = make_long_fixture();
    for (int cap : {2, 3, 5, 8, 20, 100}) {
        const auto kept = pool_keep_checkpoints(f.rows, f.is_msg, cap, 72891, 72870);
        CHECK((int) kept.size() == std::min<int>((int) f.rows.size(), cap));
        CHECK(std::is_sorted(kept.begin(), kept.end()));
        CHECK(std::adjacent_find(kept.begin(), kept.end()) == kept.end());
    }
    // No anchor and no turn start: the smallest legal cap keeps the oldest row
    // and, standing in for the anchor, the newest (commit) row.
    const auto minimal = pool_keep_checkpoints(f.rows, f.is_msg, 2, -1, -1);
    CHECK(minimal.size() == 2);
    CHECK(minimal.front() == f.rows.front());
    CHECK(minimal.back() == f.rows.back());
    // Message starts outlive interval rows: every interval row goes first.
    // Protected here: the oldest row, the turn row (60000) and, with no
    // anchor, the newest row. Three slots remain for three message starts.
    const auto m = make_multi_message_fixture();
    const auto kept_m = pool_keep_checkpoints(m.rows, m.is_msg, 6, -1, 60000);
    for (int row : {4096, 45056, 51200, 56320, 60000, 66560})
        CHECK(std::find(kept_m.begin(), kept_m.end(), row) != kept_m.end());
    // One slot fewer: a message start has to go, and only once no interval
    // row is left to take its place.
    const auto tight = pool_keep_checkpoints(m.rows, m.is_msg, 5, -1, 60000);
    int messages = 0;
    for (int row : tight) {
        CHECK(row == 4096 || row >= 45056); // no interval row survives
        if (row == 45056 || row == 51200 || row == 56320) ++messages;
    }
    CHECK(messages == 2);
}

static void test_checkpoint_resume_row() {
    // Mirrors pool_resume_row: the largest checkpoint at or before
    // min(lcp, rows, cap).
    const std::vector<int> rows = {4096, 8192, 12288, 72870};
    int keep = -1;
    CHECK(pool_checkpoint_row(rows, 50000, 72895, 72895, keep) && keep == 12288);
    CHECK(pool_checkpoint_row(rows, 72892, 72895, 72891, keep) && keep == 72870);
    // A cap below every checkpoint has nothing to resume from.
    CHECK(!pool_checkpoint_row(rows, 72892, 72895, 100, keep));
    CHECK(keep == -1);
    // A checkpoint beyond the stored rows is not a candidate.
    CHECK(!pool_checkpoint_row({90000}, 90000, 80000, 90000, keep));
}

static void test_route_switches_when_worthwhile() {
    const pool_policy_limits limits;
    pool_live live;
    live.rows = 8000;
    live.lcp = 3;
    live.row = -1;
    live.valuable = false; // the live conversation is not worth keeping

    pool_entry parked;
    parked.rows = 9000;
    parked.lcp = 8500;
    parked.row = 8500;
    parked.valuable = false;

    const auto plan = pool_route(live, {parked}, limits);
    CHECK(plan.action == pool_action::switch_in);
    CHECK(plan.restore == 0);
    CHECK(plan.resume == 8500);
    CHECK(!plan.park_live);
    CHECK(!plan.fork);

    // The gain must clear min_gain, or the request stays where it is.
    pool_entry marginal = parked;
    marginal.row = limits.min_gain - 1; // live.row is -1, so the gain is the row itself
    CHECK(pool_route(live, {marginal}, limits).action == pool_action::none);
}

static void test_route_parks_an_abandoned_branch() {
    const pool_policy_limits limits;
    pool_live live;
    live.rows = 70000;
    live.lcp = 40000;
    live.row = 40960;
    live.valuable = true; // this conversation must survive the request

    pool_entry parked;
    parked.rows = 3000;
    parked.lcp = 1000;
    parked.row = 1000; // too little gain to switch

    const auto plan = pool_route(live, {parked}, limits);
    CHECK(plan.action == pool_action::park_only);
    CHECK(plan.park_live);
    CHECK(plan.resume == 40960);
    CHECK(std::string(plan.reason) == "branch");

    // With nothing live to park, the same request is a plain miss.
    pool_live empty;
    empty.rows = 70000;
    empty.row = -1;
    empty.valuable = false;
    CHECK(pool_route(empty, {parked}, limits).action == pool_action::none);
}

static void test_route_fork_when_the_entry_outlives_the_prefix() {
    const pool_policy_limits limits;
    pool_live live;
    live.rows = 8000;
    live.lcp = 3;
    live.row = -1;
    live.valuable = true; // must survive, so the entry is shared, not consumed

    pool_entry parked;
    parked.rows = 9000;
    parked.lcp = 8500;
    parked.row = 8500;
    parked.valuable = true;

    const auto plan = pool_route(live, {parked}, limits);
    CHECK(plan.action == pool_action::switch_in);
    CHECK(plan.park_live);
    CHECK(plan.fork);
    CHECK(std::string(plan.reason) == "switch");
}

static void test_route_picks_the_best_entry() {
    const pool_policy_limits limits;
    pool_live live;
    live.rows = 100;
    live.lcp = 0;
    live.row = -1;
    live.valuable = false;

    pool_entry a; a.rows = 5000; a.lcp = 4000; a.row = 4000;
    pool_entry b; b.rows = 9000; b.lcp = 8000; b.row = 8000; // best
    pool_entry c; c.rows = 6000; c.lcp = 1000; c.row = 1000;

    const auto plan = pool_route(live, {a, b, c}, limits);
    CHECK(plan.restore == 1);
    CHECK(plan.resume == 8000);
}

int main() {
    test_incident_deep_branch_is_parked();
    test_rewritten_tail_is_not_parked();
    test_incident_checkpoint_spread();
    test_keep_is_bounded_and_ordered();
    test_checkpoint_resume_row();
    test_route_switches_when_worthwhile();
    test_route_parks_an_abandoned_branch();
    test_route_fork_when_the_entry_outlives_the_prefix();
    test_route_picks_the_best_entry();
    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS conversation pool routing policy: incidents, spread, switch/park/fork\n");
    return 0;
}

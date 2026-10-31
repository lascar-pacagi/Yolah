#ifndef SEARCH_TRACE_H
#define SEARCH_TRACE_H
// The recorder of the search debugger (test/search_trace_main.cpp,
// ui/search_debugger.html): what MinMaxNNUE_DevPlayer's search does, event by
// event, with the source line of each event. Only compiled into the
// search_trace executable (YOLAH_SEARCH_TRACE, see the TR / TRET macros of
// minmax_nnue_dev_player.cpp); the players of every other executable have no
// hook at all.
//
// Events are int32 words: [type | nb_args << 8, line, args...]. The types are
// listed in EVENT_NAMES (the page reads their names from the trace's header).
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct SearchTracer {
    enum Event : int32_t {
        SEARCH = 1,     // a search starts                     (root ply, nb moves)
        ITER,           // root_search called by the deepening  (depth, alpha, beta, attempt, δ: current half width)
        ITER_RESULT,    // its result                           (depth, value, move, outcome 0 ok 1 fail low 2 fail high 3 stopped)
        PV,             // the principal variation from the table (n, moves...)
        ROOT_ORDER,     // root moves after the iteration        (n, (move, nodes)...)
        ENTER,          // negamax / root_search starts          (root?, alpha, beta, depth, ply, move to it, hash lo, hash hi, nnue)
        RETURN,         // it returns                            (value)
        TT_PROBE,       // table probe                           (found, depth, lower, upper, move)
        EVAL,           // evaluate()                            (value, from the cache?)
        RFP,            // reverse futility test                 (static, margin, beta, cut?)
        MPC,            // Multi-ProbCut try                     (direction 1 high 2 low, d', margin, bound)
        MPC_RESULT,     //                                       (direction, value, cut?)
        PROXY,          // proxy cut witness (or ProbCut move)   (move, bound, depth)
        PROXY_RESULT,   //                                       (move, value, cut?)
        GEN,            // move generation                       (stage 1 = table's move only, 2 = all; nb moves)
        SCORES,         // move scores                           (n, (move, score, territory)...)
        MOVE,           // a move of the loop                    (i, move, score, reduction, tail reduction)
        CALL,           // search_move's child searches          (kind 1 reduced 2 null window 3 re-search 4 full, alpha, beta, depth)
        CALL_RESULT,    //                                       (kind, value for the node)
        MOVE_DONE,      //                                       (i, move, value, best, alpha)
        LMP,            // late move pruning stops the loop      (i, n)
        CUTOFF,         // beta cutoff                           (i, move, value, beta)
        KILLER,         //                                       (ply, killer1, killer2 after)
        HISTORY,        //                                       (player, move, before, after, bonus)
        COUNTER,        //                                       (previous move, countermove after)
        TT_STORE,       // tt_store                              (depth, bound, value, move; old: hash lo, hi, depth, bound, value, move;
                        //                                        new: depth, bound, value, move)
        CHILD_EVALS,    // the network's value of every child    (n, (move, value for the node)...)
        STOPPED,        // the recording is full: the search was stopped
        YT_STORE,       // tt_store, Yolah table                 (depth, alpha, beta, value, move; old: hash lo, hi, depth
                        //                                        (−1: empty), lower, upper, move; new: depth, lower, upper, move)
        NB_EVENTS
    };
    static constexpr const char* EVENT_NAMES[NB_EVENTS] = {
        "", "SEARCH", "ITER", "ITER_RESULT", "PV", "ROOT_ORDER", "ENTER", "RETURN", "TT_PROBE", "EVAL", "RFP", "MPC",
        "MPC_RESULT", "PROXY", "PROXY_RESULT", "GEN", "SCORES", "MOVE", "CALL", "CALL_RESULT", "MOVE_DONE", "LMP",
        "CUTOFF", "KILLER", "HISTORY", "COUNTER", "TT_STORE", "CHILD_EVALS", "STOPPED", "YT_STORE"};

    std::vector<int32_t> words;
    size_t max_words = size_t(1) << 26;    // 256 MB of events, then the search is stopped
    std::atomic_bool* stop = nullptr;      // the player's clock flag
    bool full = false;
    int child_evals_depth = 1;             // CHILD_EVALS at nodes of at least this depth (0 = never)
    // Scratch for SCORES: the territory term of each move, filled by score_moves.
    int territory[128]{};
    // The full hash of the positions stored in the table, by their 32-bit key
    // (what the table compares): to name the position an entry was evicted for.
    std::unordered_map<uint32_t, uint64_t> hash_of_key;
    // The same for the Yolah table (SearchTable), by its key32 << 16 | key16.
    std::unordered_map<uint64_t, uint64_t> hash_of_ykey;

    bool on() const { return !full; }
    void push(int32_t w) { words.push_back(w); }
    template <class... A> void ev(int line, Event type, A... args) {
        if (full) return;
        push(int32_t(type) | int32_t(sizeof...(A) << 8));
        push(line);
        (push(int32_t(args)), ...);
        check();
    }
    // A variable-length event: header, line, then the words of `body`.
    void ev_list(int line, Event type, const std::vector<int32_t>& body) {
        if (full) return;
        push(int32_t(type) | int32_t(body.size() << 8));
        push(line);
        for (int32_t w : body) push(w);
        check();
    }
    void check() {
        if (words.size() >= max_words && !full) {
            full = true;
            push(int32_t(STOPPED));
            push(0);
            if (stop) stop->store(true);
        }
    }
};

#endif

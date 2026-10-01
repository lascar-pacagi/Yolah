#include "minmax_nnue_baseline_player.h"
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "zobrist.h"
#include <utility>

using std::cout, std::endl;

MinMaxNNUE_BaselinePlayer::MinMaxNNUE_BaselinePlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth,
                                                     uint8_t late_move_reduction, const std::string& nnue_q_parameters_filename,
                                                     bool verbose)
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction),
      nnue_q_parameters_filename(nnue_q_parameters_filename), verbose(verbose) {
    nnue.load(nnue_q_parameters_filename);
}

MinMaxNNUE_BaselinePlayer::Result MinMaxNNUE_BaselinePlayer::search(const Yolah& yolah, uint8_t max_depth, uint64_t microseconds) {
    const auto start = std::chrono::steady_clock::now();
    stop = false;
    // The clock: sets `stop` after `microseconds`, unless the search is over
    // before (then the jthread's destructor wakes it up at once).
    std::mutex m;
    std::condition_variable_any cv;
    std::jthread clock;
    if (microseconds > 0) {
        clock = std::jthread([&](std::stop_token st) {
            std::unique_lock lock(m);
            if (!cv.wait_for(lock, st, std::chrono::microseconds(microseconds), [] { return false; })) {
                stop = true;
            }
        });
    }
    table.new_search();
    Search s;
    nnue.init(yolah, s.acc);
    iterative_deepening(yolah, s, max_depth);
    if (clock.joinable()) {
        clock.request_stop();
        clock.join();
    }
    Result r;
    r.move = s.move;
    r.value = s.value;
    r.depth = s.depth;
    r.nb_nodes = s.nb_nodes;
    r.nb_hits = s.nb_hits;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}

void MinMaxNNUE_BaselinePlayer::clear_table() {
    table.clear(1);
}

Move MinMaxNNUE_BaselinePlayer::play(Yolah yolah) {
    Result r = search(yolah, 63, thinking_time);
    if (verbose) {
        cout << "##########\n";
        cout << "depth  : " << int(r.depth) << '\n';
        cout << "value  : " << r.value << '\n';
        cout << "# nodes: " << r.nb_nodes << '\n';
        cout << "# hits : " << r.nb_hits << '\n';
        cout << "tt load: " << table.load() << '\n';
        print_pv(yolah, zobrist::hash(yolah), r.depth);
        cout << '\n';
    }
    return r.move;
}

std::string MinMaxNNUE_BaselinePlayer::info() {
    return "minmax nnue baseline player (MinMaxNNUE_QuantizedPlayer's search on one thread: "
           "transposition table + late move reduction + killer)";
}

int16_t MinMaxNNUE_BaselinePlayer::negamax(Yolah& yolah, Search& s, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth) {
    ++s.nb_nodes;
    if (yolah.game_over()) {
        int16_t score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (found) s.nb_hits++;
    if (found && entry->depth() >= depth) {
        int16_t v = entry->value();
        if (entry->bound() == BOUND_EXACT) {
            return v;
        }
        if (entry->bound() == BOUND_LOWER) {
            if (v >= beta) return v;
            alpha = std::max(alpha, v);
        }
        if (entry->bound() == BOUND_UPPER) {
            if (v <= alpha) return v;
            beta = std::min(beta, v);
        }
    }
    if (depth <= 0) {
        int16_t v = nnue.value(s.acc, yolah.current_player()) * heuristic::MAX_VALUE;
        table.update(hash, v, BOUND_EXACT, 0);
        return v;
    }
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, s, hash, moves);
    Bound b = BOUND_UPPER;
    Move  move = Move::none();
    auto player = yolah.current_player();
    for (size_t i = 0; i < moves.size(); i++) {
        Move m = moves[i];
        if (i >= nb_moves_at_full_depth) {
            nnue.play(yolah.current_player(), m, s.acc);
            yolah.play(m);
            int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            nnue.undo(yolah.current_player(), m, s.acc);
            if (v <= alpha) continue;
        }
        nnue.play(yolah.current_player(), m, s.acc);
        yolah.play(m);
        int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
        yolah.undo(m);
        nnue.undo(yolah.current_player(), m, s.acc);
        if (v >= beta) {
            table.update(hash, v, BOUND_LOWER, depth, m);
            s.killer2[yolah.nb_plies()] = s.killer1[yolah.nb_plies()];
            s.killer1[yolah.nb_plies()] = m;
            return v;
        }
        if (v > alpha) {
            alpha = v;
            b = BOUND_EXACT;
            move = m;
        }
        if (stop) {
            return 0;
        }
    }
    table.update(hash, alpha, b, depth, move);
    s.killer2[yolah.nb_plies()] = s.killer1[yolah.nb_plies()];
    s.killer1[yolah.nb_plies()] = move;
    return alpha;
}

int16_t MinMaxNNUE_BaselinePlayer::root_search(Yolah& yolah, Search& s, uint64_t hash, int8_t depth, Move& res) {
    res = Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    int16_t alpha = -std::numeric_limits<int16_t>::max();
    int16_t beta  = std::numeric_limits<int16_t>::max();
    sort_moves(yolah, s, hash, moves);
    auto player = yolah.current_player();
    for (size_t i = 0; i < moves.size(); i++) {
        Move m = moves[i];
        if (i >= nb_moves_at_full_depth) {
            nnue.play(yolah.current_player(), m, s.acc);
            yolah.play(m);
            int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            nnue.undo(yolah.current_player(), m, s.acc);
            if (v <= alpha) continue;
        }
        nnue.play(yolah.current_player(), m, s.acc);
        yolah.play(m);
        int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
        yolah.undo(m);
        nnue.undo(yolah.current_player(), m, s.acc);
        if (v > alpha) {
            alpha = v;
            res = m;
        }
        if (stop) {
            return 0;
        }
    }
    table.update(hash, alpha, BOUND_EXACT, depth, res);
    return alpha;
}

void MinMaxNNUE_BaselinePlayer::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
    Move tmp[Yolah::MAX_NB_MOVES];
    size_t nb_moves = moves.size();
    Move best = table.get_move(hash);
    Move killer_move1 = s.killer1[yolah.nb_plies()];
    Move killer_move2 = s.killer2[yolah.nb_plies()];
    Move b = Move::none();
    Move k1 = Move::none();
    Move k2 = Move::none();
    size_t n = 0;
    for (size_t i = 0; i < nb_moves; i++) {
        Move m = moves[i];
        if (m == best) b = m;
        else if (m == killer_move1) k1 = m;
        else if (m == killer_move2) k2 = m;
        else tmp[n++] = m;
    }
    size_t i = 0;
    if (b != Move::none())  moves[i++] = b;
    if (k1 != Move::none()) moves[i++] = k1;
    if (k2 != Move::none()) moves[i++] = k2;
    for (size_t j = 0; j < n; j++) {
        moves[i++] = tmp[j];
    }
}

void MinMaxNNUE_BaselinePlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

void MinMaxNNUE_BaselinePlayer::iterative_deepening(Yolah yolah, Search& s, uint8_t max_depth) {
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    uint8_t depth = 0;
    int16_t value = 0;
    for (uint8_t d = 1; d <= max_depth && d < 64; d++) {
        Move m;
        auto v = root_search(yolah, s, hash, d, m);
        if (stop) {
            break;
        }
        res = m;
        depth = d;
        value = v;
    }
    s.depth = depth;
    s.value = value;
    s.move = res;
}

json MinMaxNNUE_BaselinePlayer::config() {
    json j;
    j["name"] = "MinMaxNNUE_BaselinePlayer";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    j["weights"] = nnue_q_parameters_filename;
    j["verbose"] = verbose;
    return j;
}

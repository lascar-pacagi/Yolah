#include "minmax_player_v9.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayerV9::MinMaxPlayerV9(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, 
                               uint8_t late_move_reduction, size_t nb_threads, heuristic_eval h) 
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction), 
      pool(static_cast<BS::concurrency_t>(nb_threads)), heuristic(h) {
}

Move MinMaxPlayerV9::play(Yolah yolah) {
    this->stop = false;
    std::jthread clock([this]{
        std::this_thread::sleep_for(std::chrono::microseconds(this->thinking_time));
        this->stop = true;
    });
    table.new_search();
    std::vector<Search> results(pool.get_thread_count());
    BS::multi_future<void> futures = pool.submit_sequence<size_t>(0, pool.get_thread_count(), [&](size_t i) { 
        iterative_deepening(yolah, results[i]);
    });
    futures.get();
    uint8_t depth = 0;
    Move res = Move::none();
    for (Search& s : results) {
        if (s.depth > depth) {
            depth = s.depth;
            res = s.move;
        }
    }
    return res;
}

std::string MinMaxPlayerV9::info() {
    return "minmax player v9 (v2 + lazy SMP)";
}

int16_t MinMaxPlayerV9::negamax(Yolah& yolah, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth) {
    //nb_nodes++;
    if (yolah.game_over()) {
        int16_t score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    if (depth <= 0) {
        return heuristic(yolah.current_player(), yolah);
    }    
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    //if (found) nb_hits++;
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
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, hash, moves);
    Bound b = BOUND_UPPER;
    Move  move = Move::none();
    auto player = yolah.current_player();
    for (size_t i = 0; i < moves.size(); i++) {        
        Move m = moves[i];
        if (i >= nb_moves_at_full_depth) {
            yolah.play(m);
            int16_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v <= alpha) continue;
        }
        yolah.play(m);
        int16_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
        yolah.undo(m);
        if (v >= beta) {
            table.update(hash, v, BOUND_LOWER, depth, m);
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
    return alpha;
}
    
int16_t MinMaxPlayerV9::root_search(Yolah& yolah, uint64_t hash, int8_t depth, Move& res) {
    res = Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    int16_t alpha = -std::numeric_limits<int16_t>::max();
    int16_t beta  = std::numeric_limits<int16_t>::max();
    sort_moves(yolah, hash, moves);
    auto player = yolah.current_player();
    for (size_t i = 0; i < moves.size(); i++) {     
        Move m = moves[i];
        if (i >= nb_moves_at_full_depth) {
            yolah.play(m);
            int16_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v <= alpha) continue;
        }   
        yolah.play(m);
        int16_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
        yolah.undo(m);
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
    
void MinMaxPlayerV9::sort_moves(Yolah& yolah, uint64_t hash, Yolah::MoveList& moves) {
    std::vector<std::pair<int16_t, Move>> tmp;
    size_t nb_moves = moves.size();
    auto player = yolah.current_player();
    Move best = table.get_move(hash);
    for (size_t i = 0; i < nb_moves; i++) {
        Move m = moves[i];
        if (best == m) {
            tmp.emplace_back(std::numeric_limits<int16_t>::max(), best);
        } else {
            yolah.play(m);
            tmp.emplace_back(heuristic(player, yolah), moves[i]);
            yolah.undo(m);
        }
    }
    std::sort(begin(tmp), end(tmp), [](const auto& p1, const auto& p2) {
        return p1.first > p2.first;
    });
    for (size_t i = 0; i < nb_moves; i++) {
        moves[i] = tmp[i].second;
    }
}

void MinMaxPlayerV9::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

void MinMaxPlayerV9::iterative_deepening(Yolah yolah, Search& search) {    
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    uint8_t depth = 0;
    int16_t value = 0;
    for (uint8_t d = 1; d < 64; d++) {
        Move m;
        auto v = root_search(yolah, hash, d, m);      
        if (stop) {
            break;
        }
        res = m;
        depth = d;
        value = v;
        // cout << "depth  : " << int(depth) << '\n';
        // cout << "value  : " << value << '\n';
        // cout << "move   : " << m << '\n';
        // cout << "# nodes: " << nb_nodes << '\n';
        // cout << "# hits : " << nb_hits << '\n';
        // print_pv(yolah, hash, depth);
        // cout << std::endl;
    }
    search.depth = depth;
    search.value = value;
    search.move = res;
}

json MinMaxPlayerV9::config() {
    json j;
    j["name"] = "MinMaxPlayerV9";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    if (pool.get_thread_count() == std::thread::hardware_concurrency()) {
        j["nb threads"] = "hardware concurrency";
    } else {
        j["nb threads"] = pool.get_thread_count();
    }
    return j;
}

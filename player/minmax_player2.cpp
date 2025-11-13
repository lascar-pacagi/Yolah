#include "minmax_player2.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayer2::MinMaxPlayer2(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth,
                            uint8_t late_move_reduction, size_t nb_threads, heuristic_eval h)
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction),
      pool(static_cast<BS::concurrency_t>(nb_threads)), heuristic(h) {
}

Move MinMaxPlayer2::play(Yolah yolah) {
    this->stop = false;
    std::jthread clock([this]{
        std::this_thread::sleep_for(std::chrono::microseconds(this->thinking_time));
        this->stop = true;
    });
    // Yolah::MoveList moves;
    // yolah.blocking_moves(moves);
    // cout << "blocking moves: ";
    // for (Move m : moves) {
    //     cout << m << ' ';
    // }
    // cout << '\n';
    table.new_search();
    std::vector<Search> results(pool.get_thread_count());
    BS::multi_future<void> futures = pool.submit_sequence<size_t>(0, pool.get_thread_count(), [&](size_t i) { 
        iterative_deepening(yolah, results[i]);
    });
    futures.get();
    uint8_t depth = 0;
    int16_t value = 0;
    Move res = Move::none();
    size_t nb_nodes = 0;
    size_t nb_hits = 0;
    // cout << "##########\n";
    // Yolah::MoveList moves;
    // yolah.encircling_or_escaping_moves(yolah.current_player(), moves);
    // for (Move m : moves) {
    //     cout << m << ' ';
    // }
    // cout << '\n';
    for (Search& s : results) {
        // cout << int(s.depth) << ' ' << s.move << '\n';
        if (s.depth > depth) {
            depth = s.depth;
            res = s.move;
            value = s.value;
            nb_nodes += s.nb_nodes;
            nb_hits += s.nb_hits;            
        }
    }
    // cout << "##########\n";
    // cout << "depth  : " << int(depth) << '\n';
    // cout << "value  : " << value << '\n';
    // cout << "# nodes: " << nb_nodes << '\n';
    // cout << "# hits : " << nb_hits << '\n';
    // cout << "tt load: " << table.load() << '\n';
    // print_pv(yolah, zobrist::hash(yolah), depth);
    // cout << '\n';
    // cout << heuristic::eval(yolah.current_player(), yolah) << endl;
    return res;
}

std::string MinMaxPlayer2::info() {
    return "minmax player 2 (transposition table + late move reduction + killer + lazy SMP)";
}

namespace {
    // Approximation, you can get false negatives but no false positives
    template<int N>
    bool can_move(uint64_t bb, uint64_t free) {
        auto lsb_and_pop = [](uint64_t& bitboard) {
            uint64_t last_bit = bitboard & -bitboard;
            bitboard ^= last_bit;
            return last_bit;
        };
        uint64_t bb1 = lsb_and_pop(bb);
        uint64_t bb2 = lsb_and_pop(bb);
        uint64_t bb3 = lsb_and_pop(bb);
        uint64_t bb4 = lsb_and_pop(bb);
        int count = 0;
        for (int i = 0; i < N; i++) {
            bb1 = shift_all_directions(bb1) & free;
            uint64_t bit1 = bb1 & -bb1;
            free &= ~bit1;
            bb1 = bit1;
            count += (bb1 != 0);
            
            bb2 = shift_all_directions(bb2) & free;
            uint64_t bit2 = bb2 & -bb2;
            free &= ~bit2;
            bb2 = bit2;
            count += (bb2 != 0);
            
            bb3 = shift_all_directions(bb3) & free;
            uint64_t bit3 = bb3 & -bb3;
            free &= ~bit3;
            bb3 = bit3;
            count += (bb3 != 0);
            
            bb4 = shift_all_directions(bb4) & free;
            uint64_t bit4 = bb4 & -bb4;
            free &= ~bit4;
            bb4 = bit4;
            count += (bb4 != 0);
            
            if (count >= N) return true;
        }
        return false;
    }
}

int16_t MinMaxPlayer2::negamax(Yolah& yolah, Search& s, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth, Move last_last_last_move, Move last_last_move, Move last_move) {
    ++s.nb_nodes;
    if (yolah.game_over()) {
        int16_t score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    if (yolah.current_player() == Yolah::BLACK) {
        if (last_move == Move::none()) return 1 + heuristic::MAX_VALUE; // Conservative value
    } else {
        if (last_last_last_move == Move::none() && last_move == Move::none()) return 1 + heuristic::MAX_VALUE; // Conservative value
    }
    // if (last_move == Move::none()) {
    //     if (yolah.current_player() == Yolah::BLACK 
    //         || can_move<2>(yolah.bitboard(Yolah::WHITE), yolah.free_squares())) {
    //         return 1 + heuristic::MAX_VALUE; // Conservative value
    //     }
    // }    
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
        int16_t v = heuristic(yolah.current_player(), yolah);//quiescence(yolah, s, alpha, beta, 4);//heuristic(yolah.current_player(), yolah);
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
            yolah.play(m);
            int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction, last_last_move, last_move, m);
            yolah.undo(m);
            if (v <= alpha) continue;
        }
        yolah.play(m);
        int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - 1, last_last_move, last_move, m);
        yolah.undo(m);
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
    
int16_t MinMaxPlayer2::root_search(Yolah& yolah, Search& s, uint64_t hash, int8_t depth, Move& res) {
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
            yolah.play(m);
            int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction, Move(-1), Move(-1), m);
            yolah.undo(m);
            if (v <= alpha) continue;
        }   
        yolah.play(m);
        int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - 1, Move(-1), Move(-1), m);
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

void MinMaxPlayer2::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
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

void MinMaxPlayer2::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

void MinMaxPlayer2::iterative_deepening(Yolah yolah, Search& s) {    
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    uint8_t depth = 0;
    int16_t value = 0;
    for (uint8_t d = 1; d < 64; d++) {
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

json MinMaxPlayer2::config() {
    json j;
    j["name"] = "MinMaxPlayer2";
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

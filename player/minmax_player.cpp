#include "minmax_player.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayer::MinMaxPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, 
                           uint8_t late_move_reduction, size_t nb_threads, heuristic_eval h) 
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction), 
      pool(static_cast<BS::concurrency_t>(nb_threads)), heuristic(h) {
}

Move MinMaxPlayer::play(Yolah yolah) {
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
    cout << "##########\n";
    cout << "depth  : " << int(depth) << '\n';
    cout << "value  : " << value << '\n';
    cout << "# nodes: " << nb_nodes << '\n';
    cout << "# hits : " << nb_hits << '\n';
    cout << "tt load: " << table.load() << '\n';
    print_pv(yolah, zobrist::hash(yolah), depth);
    cout << '\n';
    //cout << heuristic::eval(yolah.current_player(), yolah) << endl;
    return res;
}

std::string MinMaxPlayer::info() {
    return "minmax player (transposition table + late move reduction + killer + lazy SMP)";
}

int16_t MinMaxPlayer::quiescence(Yolah& yolah, Search& s, int16_t alpha, int16_t beta, int8_t depth) {
    ++s.nb_nodes;
    if (yolah.game_over()) {
        int16_t score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    if (depth <= 0) {
        return heuristic(yolah.current_player(), yolah);
    }
    Yolah::MoveList moves;
    yolah.contact_moves(moves);
    if (moves.size() == 0) {
        return heuristic(yolah.current_player(), yolah);
    }
    auto player = yolah.current_player();
    for (size_t i = 0; i < moves.size(); i++) {
        Move m = moves[i];
        yolah.play(m);
        int16_t v = -quiescence(yolah, s, -beta, -alpha, depth - 1);
        yolah.undo(m);
        if (v >= beta) {
            return v;
        }
        if (v > alpha) {
            alpha = v;
        }
        if (stop) {
            return 0;
        }
    }
    return alpha;
}

int16_t MinMaxPlayer::negamax(Yolah& yolah, Search& s, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth) {
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
            int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            //int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta - 1, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v <= alpha) continue;
        }
        yolah.play(m);
        int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
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
    
int16_t MinMaxPlayer::root_search(Yolah& yolah, Search& s, uint64_t hash, int8_t depth, Move& res) {
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
            int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v <= alpha) continue;
        }   
        yolah.play(m);
        int16_t v = -negamax(yolah, s, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
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
    
// void MinMaxPlayer::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
//     std::vector<std::pair<int16_t, Move>> tmp;
//     size_t nb_moves = moves.size();
//     auto player = yolah.current_player();
//     Move best = table.get_move(hash);
//     for (size_t i = 0; i < nb_moves; i++) {
//         Move m = moves[i];
//         if (m == best) {
//             tmp.emplace_back(std::numeric_limits<int16_t>::max(), best);
//         } else if (m == s.killer1[yolah.nb_plies()]) {
//             tmp.emplace_back(std::numeric_limits<int16_t>::max() - 1, m);
//         } else if (m == s.killer2[yolah.nb_plies()]) {
//             tmp.emplace_back(std::numeric_limits<int16_t>::max() - 2, m); 
//         } else if (yolah.is_blocking_move(m)) {
//             tmp.emplace_back(std::numeric_limits<int16_t>::max() - 3, m);
//         } else {
//             yolah.play(m);
//             tmp.emplace_back(heuristic(player, yolah), moves[i]);
//             yolah.undo(m);
//         }
//     }
//     std::sort(begin(tmp), end(tmp), [](const auto& p1, const auto& p2) {
//         return p1.first > p2.first;
//     });
//     for (size_t i = 0; i < nb_moves; i++) {
//         moves[i] = tmp[i].second;
//     }
// }

// void MinMaxPlayer::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
//     Move tmp[Yolah::MAX_NB_MOVES];
//     size_t n = 0;
//     size_t nb_moves = moves.size();
//     auto player = yolah.current_player();
//     Move best = table.get_move(hash);    
//     Move k1   = s.killer1[yolah.nb_plies()];
//     Move k2   = s.killer2[yolah.nb_plies()];
//     bool best_present = false;
//     bool k1_present   = false;
//     bool k2_present   = false;
//     for (size_t i = 0; i < nb_moves; i++) {
//         Move m = moves[i];
//         if (m == best) best_present = true;
//         else if (m == k1) k1_present = true;
//         else if (m == k2) k2_present = true;
//         else tmp[n++] = m;
//     }
//     size_t i = 0;
//     if (best_present) moves[i++] = best;
//     if (k1 != best && k1_present) moves[i++] = k1;
//     if (k2 != best && k2 != k1 && k2_present) moves[i++] = k2;
//     for (size_t j = 0; j < n; j++) {
//         moves[i + j] = tmp[j];
//     }
// }

// void MinMaxPlayer::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
//     std::pair<int16_t, Move> tmp[Yolah::MAX_NB_MOVES];
//     size_t nb_moves = moves.size();
//     auto player = yolah.current_player();
//     Move best = table.get_move(hash);
//     for (size_t i = 0; i < nb_moves; i++) {
//         Move m = moves[i];
//         if (m == best) {
//             tmp[i] = { std::numeric_limits<int16_t>::max(), best };
//         } else if (m == s.killer1[yolah.nb_plies()]) {
//             tmp[i] = { std::numeric_limits<int16_t>::max() - 1, m };
//         } else if (m == s.killer2[yolah.nb_plies()]) {
//             tmp[i] = { std::numeric_limits<int16_t>::max() - 2, m }; 
//         } else if (yolah.is_blocking_move(m)) {
//             tmp[i] = { std::numeric_limits<int16_t>::max() - 3, m };
//         } else {
//             tmp[i] = { 0, moves[i] };
//         }
//     }
//     std::sort(begin(tmp), begin(tmp) + nb_moves, [](const auto& p1, const auto& p2) {
//         return p1.first > p2.first;
//     });
//     for (size_t i = 0; i < nb_moves; i++) {
//         moves[i] = tmp[i].second;
//     }
// }

void MinMaxPlayer::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
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

void MinMaxPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

void MinMaxPlayer::iterative_deepening(Yolah yolah, Search& s) {    
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

json MinMaxPlayer::config() {
    json j;
    j["name"] = "MinMaxPlayer";
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

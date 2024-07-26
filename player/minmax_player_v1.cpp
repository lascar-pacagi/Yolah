#include "minmax_player_v1.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayerV1::MinMaxPlayerV1(uint64_t microseconds, size_t tt_size_mb/*nb_bits, uint8_t R*/, heuristic_eval h) 
    : thinking_time(microseconds), table(/*nb_bits*/tt_size_mb), /*R(R),*/ heuristic(h) {
}

Move MinMaxPlayerV1::play(Yolah yolah) {
    return iterative_deepening(yolah);
}

std::string MinMaxPlayerV1::info() {
    return "minmax player v1 (+ transposition table)";
}

int16_t MinMaxPlayerV1::negamax(Yolah& yolah, uint64_t hash, int16_t alpha, int16_t beta, uint8_t depth) {
    nb_nodes++;
    if (yolah.game_over()) {
        int16_t score = yolah.score(yolah.current_player());
        return score + (score >= 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    if (depth == 0) {
        return heuristic(yolah.current_player(), yolah);
    }    
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (found) nb_hits++;
    //TranspositionTableSlow::Entry* entry = table.get_entry(hash);
    //if (entry) nb_hits++;
    if (found/*entry*/ && entry->depth() >= depth) {
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
    // if (depth >= R) {
    //     auto player = yolah.current_player();
    //     yolah.play(Move::none());
    //     int32_t v = -negamax(yolah, zobrist::update(hash, player, Move::none()), -beta, -beta + 1, depth - R);
    //     yolah.undo(Move::none());
    //     if (v >= beta) {
    //         table.update(hash, v, BOUND_LOWER, depth, Move::none());
    //         return v;
    //     }
    // }
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, hash, moves);
    Bound b = BOUND_UPPER;
    Move  move = Move::none();
    auto player = yolah.current_player();
    for (const Move& m : moves) {        
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
    //entry->save(hash, alpha, b, depth, move, table.generation());
    return alpha;
}
    
int16_t MinMaxPlayerV1::root_search(Yolah& yolah, uint64_t hash, uint8_t depth, Move& res) {
    res = Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    int16_t alpha = -std::numeric_limits<int16_t>::max();
    int16_t beta  = std::numeric_limits<int16_t>::max();
    sort_moves(yolah, hash, moves);
    auto player = yolah.current_player();
    for (const Move& m : moves) {        
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
    
void MinMaxPlayerV1::sort_moves(Yolah& yolah, uint64_t hash, Yolah::MoveList& moves) {
    std::vector<std::pair<int16_t, Move>> tmp;
    size_t nb_moves = moves.size();
    auto player = yolah.current_player();
    Move best = table.get_move(hash);
    for (size_t i = 0; i < nb_moves; i++) {
        if (best == moves[i]) {
            tmp.emplace_back(std::numeric_limits<int16_t>::max(), best);
        } else {
            yolah.play(moves[i]);
            tmp.emplace_back(heuristic(player, yolah), moves[i]);
            yolah.undo(moves[i]);
        }
    }
    std::sort(begin(tmp), end(tmp), [](const auto& p1, const auto& p2) {
        return p1.first > p2.first;
    });
    for (size_t i = 0; i < nb_moves; i++) {
        moves[i] = tmp[i].second;
    }
}

// void MinMaxPlayerV1::sort_moves(Yolah& yolah, uint64_t hash, Yolah::MoveList& moves) {
//     std::vector<std::pair<int32_t, Move>> tmp;
//     size_t nb_moves = moves.size();
//     auto current_player = yolah.current_player();
//     Move best = table.get_move(hash);
//     for (size_t i = 0; i < nb_moves; i++) {
//         if (best == moves[i]) {
//             tmp.emplace_back(std::numeric_limits<int32_t>::max(), best);
//         } else {
//             yolah.play(moves[i]);
//             TranspositionTableSlow::Entry* entry = table.get_entry(hash);
//             if (entry && entry->bound == BOUND_EXACT) {
//                 tmp.emplace_back(entry->value, moves[i]);
//             } else {
//                 tmp.emplace_back(heuristic(current_player, yolah), moves[i]);
//             }     
//             yolah.undo(moves[i]);
//         }
//     }
//     std::sort(begin(tmp), end(tmp), [](const auto& p1, const auto& p2) {
//         return p1.first > p2.first;
//     });
//     for (size_t i = 0; i < nb_moves; i++) {
//         moves[i] = tmp[i].second;
//     }
// }

void MinMaxPlayerV1::print_pv(Yolah yolah, uint64_t hash, uint8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

// void MinMaxPlayerV1::print_pv(Yolah yolah, uint64_t hash, uint8_t depth) {
//     if (yolah.game_over() || depth == 0) return;
//     TranspositionTableSlow::Entry* entry = table.get_entry(hash);
//     if (!entry) return;
//     auto player = yolah.current_player();
//     cout << entry->move << ' ';
//     yolah.play(entry->move);
//     print_pv(yolah, zobrist::update(hash, player, entry->move), depth - 1);
// }

Move MinMaxPlayerV1::iterative_deepening(Yolah& yolah) {
    this->stop = false;
    std::jthread clock([this]{
        std::this_thread::sleep_for(std::chrono::microseconds(this->thinking_time));
        this->stop = true;
    });
    table.new_search();
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    for (uint8_t depth = 1; depth < 64; depth++) {
        nb_nodes = 0;
        nb_hits  = 0;
        Move m;
        auto value = root_search(yolah, hash, depth, m);      
        if (stop) {
            break;
        }
        res = m;
        // cout << "depth  : " << int(depth) << '\n';
        // cout << "value  : " << value << '\n';
        // cout << "move   : " << m << '\n';
        // cout << "# nodes: " << nb_nodes << '\n';
        // cout << "# hits : " << nb_hits << '\n';
        // print_pv(yolah, hash, depth);
        // cout << std::endl;
    }
    return res;
}

json MinMaxPlayerV1::config() {
    json j;
    j["name"] = "MinMaxPlayerV1";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    return j;
}

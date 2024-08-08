#include "minmax_player_v4.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayerV4::MinMaxPlayerV4(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, 
                               uint8_t late_move_reduction, heuristic_eval h) 
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction), heuristic(h) {
}

Move MinMaxPlayerV4::play(Yolah yolah) {
    return iterative_deepening(yolah);
}

std::string MinMaxPlayerV4::info() {
    return "minmax player v4 (v2 + aspiration window)";
}

int32_t MinMaxPlayerV4::negamax(Yolah& yolah, uint64_t hash, int32_t alpha, int32_t beta, int8_t depth) {
    nb_nodes++;
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
    if (found) nb_hits++;
    if (found && entry->depth() >= depth) {
        int16_t v = entry->value();
        if (entry->bound() == BOUND_EXACT) {
            return v;
        }
        if (entry->bound() == BOUND_LOWER) {
            if (v >= beta) return v;
            if (v > alpha) alpha = v;
        }
        if (entry->bound() == BOUND_UPPER) {
            if (v <= alpha) return v;
            if (v < beta) beta = v;
        }
    }
    int32_t value = std::numeric_limits<int16_t>::min();
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
            int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v > value) value = v;
            if (v <= alpha) {
                continue;
            }
        }
        yolah.play(m);
        int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
        yolah.undo(m);
        if (v > value) value = v;
        if (v > alpha) {
            alpha = v;
            b = BOUND_EXACT;
            move = m;
            table.set_move(hash, move);            
        }
        if (v >= beta) {
            table.update(hash, v, BOUND_LOWER, depth, m);
            return v;
        }     
        if (stop) {
            return 0;
        }
    }
    entry->save(hash, value, b, depth, move, table.generation());
    return value;
}
    
// int32_t MinMaxPlayerV4::root_search(Yolah& yolah, uint64_t hash, int32_t alpha, int32_t beta, int8_t depth, Move& res) {
//     res = Move::none();
//     Yolah::MoveList moves;
//     yolah.moves(moves);
//     sort_moves(yolah, hash, moves);
//     auto player = yolah.current_player();
//     for (size_t i = 0; i < moves.size(); i++) {
//         Move m = moves[i];
//         if (i >= nb_moves_at_full_depth) {
//             yolah.play(m);
//             int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
//             yolah.undo(m);
//             if (v <= alpha) continue;
//         }   
//         yolah.play(m);
//         int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
//         yolah.undo(m);
//         if (v >= beta) {
//             table.update(hash, v, BOUND_LOWER, depth, m);
//         }
//         if (v > alpha) {
//             alpha = v;
//             res = m;
//         }
//         if (stop) {
//             return 0;
//         }
//     }
//     table.update(hash, alpha, BOUND_EXACT, depth, res);
//     return alpha;
// }
    
void MinMaxPlayerV4::sort_moves(Yolah& yolah, uint64_t hash, Yolah::MoveList& moves) {
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

void MinMaxPlayerV4::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

Move MinMaxPlayerV4::iterative_deepening(Yolah& yolah) {
    this->stop = false;
    std::jthread clock([this]{
        std::this_thread::sleep_for(std::chrono::microseconds(this->thinking_time));
        this->stop = true;
    });
    static constexpr int32_t aspiration[] = {100, 500, 3000, 300000};    
    table.new_search();
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    int32_t value = 0;//negamax(yolah, hash, -1000, 1000, 1);         
    for (uint8_t depth = 1; depth < 64; depth++) {
        //cout << "depth: " << int(depth) << endl;
        nb_nodes = 0;
        nb_hits  = 0;
        Move m = Move::none();
        size_t lo_index = 0;
        size_t hi_index = 0;
        for (;;) {
            int32_t alpha = value - aspiration[lo_index];
            int32_t beta = value + aspiration[hi_index];
            value = negamax(yolah, hash, alpha, beta, depth); 
            //root_search(yolah, hash, alpha, beta, depth, m);
            //cout << '[' << alpha << ',' << beta << "] " << value << endl;
            if (stop) {
                break;
            }
            if (value <= alpha) {
                lo_index++;
            } else if (value >= beta) {
                hi_index++;
            } else {
                break;
            }
        }
        if (stop) {
            break;
        }
        res = table.get_move(hash);
        // cout << "depth  : " << int(depth) << '\n';
        // cout << "value  : " << value << '\n';
        // cout << "move   : " << res << '\n';
        // cout << "# nodes: " << nb_nodes << '\n';
        // cout << "# hits : " << nb_hits << '\n';
        // print_pv(yolah, hash, depth);
        // cout << std::endl;
    }
    return res;
}

json MinMaxPlayerV4::config() {
    json j;
    j["name"] = "MinMaxPlayerV4";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    return j;
}

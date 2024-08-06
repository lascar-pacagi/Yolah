#include "minmax_player_v7.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayerV7::MinMaxPlayerV7(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, 
                               uint8_t late_move_reduction, heuristic_eval h) 
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction), heuristic(h) {
}

Move MinMaxPlayerV7::play(Yolah yolah) {
    return iterative_deepening(yolah);
}

std::string MinMaxPlayerV7::info() {
    return "minmax player v7 (v2 + history)";
}

int16_t MinMaxPlayerV7::negamax(Yolah& yolah, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth) {
    nb_nodes++;
    if (yolah.game_over()) {
        int16_t score = yolah.score(yolah.current_player());
        return score + (score >= 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
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
            history[m.from_sq()][m.to_sq()] += 200 * int16_t(depth);
            history[m.from_sq()][m.to_sq()] = std::min(history[m.from_sq()][m.to_sq()], HISTORY_MAX);
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
    entry->save(hash, alpha, b, depth, move, table.generation());
    // history[move.from_sq()][move.to_sq()] += depth * depth;
    // history[move.from_sq()][move.to_sq()] = std::max(history[move.from_sq()][move.to_sq()], HISTORY_MAX);    
    return alpha;
}
    
int16_t MinMaxPlayerV7::root_search(Yolah& yolah, uint64_t hash, int8_t depth, Move& res) {
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
    // history[res.from_sq()][res.to_sq()] += depth * depth;
    // history[res.from_sq()][res.to_sq()] = std::max(history[res.from_sq()][res.to_sq()], HISTORY_MAX);
    return alpha;
}
    
void MinMaxPlayerV7::sort_moves(Yolah& yolah, uint64_t hash, Yolah::MoveList& moves) {
    std::vector<std::pair<int16_t, Move>> tmp;
    size_t nb_moves = moves.size();
    auto player = yolah.current_player();
    Move best = table.get_move(hash);
    for (size_t i = 0; i < nb_moves; i++) {
        Move m = moves[i];
        if (m == best) {
            tmp.emplace_back(std::numeric_limits<int16_t>::max(), best);
        } else {
            yolah.play(m);
            //cout << history[m.from_sq()][m.to_sq()] << endl;
            tmp.emplace_back(heuristic(player, yolah) + history[m.from_sq()][m.to_sq()], moves[i]);
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

void MinMaxPlayerV7::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

Move MinMaxPlayerV7::iterative_deepening(Yolah& yolah) {
    this->stop = false;
    std::jthread clock([this]{
        std::this_thread::sleep_for(std::chrono::microseconds(this->thinking_time));
        this->stop = true;
    });
    table.new_search();
    std::memset(history, 0, SQUARE_NB * SQUARE_NB * sizeof(int16_t));
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

json MinMaxPlayerV7::config() {
    json j;
    j["name"] = "MinMaxPlayerV7";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    return j;
}

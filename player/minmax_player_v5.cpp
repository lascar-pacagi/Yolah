#include "minmax_player_v5.h"
#include <thread>
#include <chrono>
#include "zobrist.h"

using std::cout, std::endl;

MinMaxPlayerV5::MinMaxPlayerV5(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, 
                               uint8_t late_move_reduction, heuristic_eval h) 
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction), heuristic(h) {
}

Move MinMaxPlayerV5::play(Yolah yolah) {
    return iterative_deepening(yolah);
}

std::string MinMaxPlayerV5::info() {
    return "minmax player v5 (v2 + mtdbi)";
}

int32_t MinMaxPlayerV5::negamax(Yolah& yolah, uint64_t hash, int32_t alpha, int32_t beta, int8_t depth) {
    nb_nodes++;
    if (yolah.game_over()) {
        int32_t score = yolah.score(yolah.current_player());
        return score + (score >= 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    if (depth <= 0) {
        return heuristic(yolah.current_player(), yolah);
    }    
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (found) nb_hits++;
    if (found && entry->depth() >= depth) {
        int32_t v = entry->value();
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
            int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v <= alpha) continue;
        }
        yolah.play(m);
        int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
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
    entry->save(hash, alpha, b, depth, move, table.generation());
    return alpha;
}
    
int32_t MinMaxPlayerV5::root_search(Yolah& yolah, uint64_t hash, int32_t alpha, int32_t beta, int8_t depth, Move& res) {
    res = Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, hash, moves);
    auto player = yolah.current_player();
    for (size_t i = 0; i < moves.size(); i++) {     
        Move m = moves[i];
        if (i >= nb_moves_at_full_depth) {
            yolah.play(m);
            int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - late_move_reduction);
            yolah.undo(m);
            if (v <= alpha) continue;
        }   
        yolah.play(m);
        int32_t v = -negamax(yolah, zobrist::update(hash, player, m), -beta, -alpha, depth - 1);
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
    
void MinMaxPlayerV5::sort_moves(Yolah& yolah, uint64_t hash, Yolah::MoveList& moves) {
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

void MinMaxPlayerV5::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

Move MinMaxPlayerV5::iterative_deepening(Yolah& yolah) {
    this->stop = false;
    std::jthread clock([this]{
        std::this_thread::sleep_for(std::chrono::microseconds(this->thinking_time));
        this->stop = true;
    });
    table.new_search();
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    int32_t value;
    for (uint8_t depth = 2; depth < 64; depth++) {
        int32_t lo = 30;
        int32_t hi = 30;
        nb_nodes = 0;
        nb_hits  = 0;
        int32_t alpha = 0;
        Move m = Move::none();        
        while (lo < hi) {
            if (stop) {
                break;
            }
            int32_t value = root_search(yolah, hash, alpha, alpha + 1, depth, m);
            if (value <= alpha) hi = value;
            else lo = value;
            int32_t mid = lo + (hi - lo) / 2;
            alpha = mid;
        }
        if (stop) {
            break;
        }
        res = m;
        // cout << "depth  : " << int(depth) << '\n';
        // cout << "value  : " << lo << '\n';
        // cout << "move   : " << m << '\n';
        // cout << "# nodes: " << nb_nodes << '\n';
        // cout << "# hits : " << nb_hits << '\n';
        // print_pv(yolah, hash, depth);
        // cout << std::endl;
    }
    return res;
}

json MinMaxPlayerV5::config() {
    json j;
    j["name"] = "MinMaxPlayerV5";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    return j;
}


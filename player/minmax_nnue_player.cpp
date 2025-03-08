#include "minmax_nnue_player.h"
#include <thread>
#include <chrono>
#include "zobrist.h"
#include <utility>

using std::cout, std::endl;

MinMaxNNUEPlayer::MinMaxNNUEPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, 
                                    uint8_t late_move_reduction, const std::string& nnue_parameters_filename, size_t nb_threads) 
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction), 
      nnue_parameters_filename(nnue_parameters_filename),
      pool(static_cast<BS::concurrency_t>(nb_threads)) {
        nnue.load(nnue_parameters_filename);
    }

Move MinMaxNNUEPlayer::play(Yolah yolah) {
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
    for (size_t i = 0; i < results.size(); i++) {
        nnue.init(yolah, results[i].acc);
    };
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

std::string MinMaxNNUEPlayer::info() {
    return "minmax nnue player (transposition table + late move reduction + killer + lazy SMP)";
}

int16_t MinMaxNNUEPlayer::negamax(Yolah& yolah, Search& s, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth) {
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
        const auto [black_proba, draw_proba, white_proba] = nnue.output(s.acc);
        float coeff = (yolah.current_player() == Yolah::BLACK ? 1 : -1); 
        int16_t v = (coeff * black_proba - coeff * white_proba) * heuristic::MAX_VALUE; 
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
    
int16_t MinMaxNNUEPlayer::root_search(Yolah& yolah, Search& s, uint64_t hash, int8_t depth, Move& res) {
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

void MinMaxNNUEPlayer::sort_moves(Yolah& yolah, const Search& s, uint64_t hash, Yolah::MoveList& moves) {
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

void MinMaxNNUEPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
}

void MinMaxNNUEPlayer::iterative_deepening(Yolah yolah, Search& s) {    
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

json MinMaxNNUEPlayer::config() {
    json j;
    j["name"] = "MinMaxNNUEPlayer";
    j["microseconds"] = thinking_time;
    j["tt size"] = table.size();
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    if (pool.get_thread_count() == std::thread::hardware_concurrency()) {
        j["nb threads"] = "hardware concurrency";
    } else {
        j["nb threads"] = pool.get_thread_count();
    }
    j["weights"] = nnue_parameters_filename;
    return j;
}

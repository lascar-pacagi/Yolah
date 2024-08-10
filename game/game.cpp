#include "game.h"
#include "magic.h"
#include "misc.h"
#include <sstream>

using std::ostream, std::pair, std::vector, std::istream;
using std::string, std::to_string, std::stoi, std::stoull; 

pair<uint16_t, uint16_t> Yolah::score() const {
    return { black_score, white_score };
}

int16_t Yolah::score(uint8_t player) const {
    return (black_score - white_score) * ((player == WHITE) * -1 + (player == BLACK)); 
}

uint8_t Yolah::current_player() const {
    return uint8_t(ply & 1);
}

uint8_t Yolah::get(Square s) const {
    uint64_t pos = square_bb(s); 
    if (black & pos) {
        return BLACK;
    }
    if (white & pos) {
        return WHITE;
    }
    if (empty & pos) {
        return EMPTY;
    }
    return FREE;
}

uint8_t Yolah::get(File f, Rank r) const {
    return get(make_square(f, r));
}

bool Yolah::game_over() const {
    uint64_t possible = ~empty & ~black & ~white;
    return
        (shift<NORTH>(black) & possible) == 0 &&
        (shift<SOUTH>(black) & possible) == 0 &&
        (shift<EAST>(black) & possible) == 0 &&
        (shift<WEST>(black) & possible) == 0 &&
        (shift<NORTH_EAST>(black) & possible) == 0 &&
        (shift<NORTH_WEST>(black) & possible) == 0 &&
        (shift<SOUTH_EAST>(black) & possible) == 0 &&
        (shift<SOUTH_WEST>(black) & possible) == 0 &&
        (shift<NORTH>(white) & possible) == 0 &&
        (shift<SOUTH>(white) & possible) == 0 &&
        (shift<EAST>(white) & possible) == 0 &&
        (shift<WEST>(white) & possible) == 0 &&
        (shift<NORTH_EAST>(white) & possible) == 0 &&
        (shift<NORTH_WEST>(white) & possible) == 0 &&
        (shift<SOUTH_EAST>(white) & possible) == 0 &&
        (shift<SOUTH_WEST>(white) & possible) == 0;     
}

bool Yolah::valid(Move m) const {
    uint64_t pos1 = square_bb(m.from_sq());
    uint64_t pos2 = square_bb(m.to_sq());
    uint64_t possible = ~empty & ~black & ~white;
    if (current_player() == BLACK) {
        return (black & pos1) && (possible & pos2);
    }
    return (white & pos1) && (possible & pos2);
}
    
void Yolah::play(Move m) {
    if (m != Move::none()) [[likely]] {
        uint64_t pos1 = square_bb(m.from_sq());
        uint64_t pos2 = square_bb(m.to_sq());
        uint64_t white_mask = uint64_t(0xFFFFFFFFFFFFFFFF) * (ply & 1); 
        uint64_t black_mask = ~white_mask;
        black ^= (black_mask & pos1) ^ (black_mask & pos2);
        white ^= (white_mask & pos1) ^ (white_mask & pos2);
        empty ^= pos1;
        black_score += black_mask & 1;
        white_score += white_mask & 1;
    }
    ply++;
}
    
void Yolah::undo(Move m) {
    ply--;
    if (m != Move::none()) [[likely]] {
        uint64_t pos1 = square_bb(m.from_sq());
        uint64_t pos2 = square_bb(m.to_sq());    
        uint64_t white_mask = uint64_t(0xFFFFFFFFFFFFFFFF) * (ply & 1);
        uint64_t black_mask = ~white_mask;
        black ^= (black_mask & pos1) ^ (black_mask & pos2);
        white ^= (white_mask & pos1) ^ (white_mask & pos2);
        empty ^= pos1;
        black_score -= black_mask & 1;
        white_score -= white_mask & 1;
    }
}

void Yolah::moves(uint8_t player, MoveList& moves) const {
    Move* moveList = moves.moveList;
    uint64_t white_mask = uint64_t(0xFFFFFFFFFFFFFFFF) * (player == WHITE); 
    uint64_t black_mask = ~white_mask;
    uint64_t occupied = black | white | empty;
    uint64_t bb = (black_mask & black) | (white_mask & white);
    debug([&]{
        std::cout << "begin moves" << std::endl;
        std::cout << Bitboard::pretty(bb);
    });    
    while (bb) {
        Square   from = pop_lsb(bb);
        uint64_t b    = attacks_bb(from, occupied) & ~occupied;        
        debug([&]{ std::cout << Bitboard::pretty(b); });
        while (b) {
            *moveList++ = Move(from, pop_lsb(b));
            debug([&]{ std::cout << *(moveList - 1) << '\n'; });
        }
    }    
    if (moveList == moves.moveList) [[unlikely]] {
        *moveList++ = Move::none();
    }
    moves.last = moveList;
    debug([&]{
        for (Move m : moves) {
            std::cout << m << ' ';
        }
        if (moves.size()) {
            std::cout << '\n' << moves.size() << '\n';
        }
        std::cout << "end moves" << std::endl;
    });
}

void Yolah::moves(MoveList& moves) const {
    Yolah::moves(current_player(), moves);
}

void Yolah::blocking_moves(uint8_t player, MoveList& moves) const {
    Move* moveList = moves.moveList;
    uint64_t white_mask = uint64_t(0xFFFFFFFFFFFFFFFF) * (player == WHITE); 
    uint64_t black_mask = ~white_mask;
    uint64_t occupied = black | white | empty;
    uint64_t bb = (black_mask & black) | (white_mask & white);    
    while (bb) {
        Square   from = pop_lsb(bb);
        uint64_t b    = attacks_bb(from, occupied) & ~occupied;        
        while (b) {
            Move m = Move(from, pop_lsb(b));
            if (is_blocking_move(player, m)) {
                *moveList++ = m;
            }            
        }
    }
    moves.last = moveList;
}

void Yolah::blocking_moves(MoveList& moves) const {
    blocking_moves(current_player(), moves);
}

bool Yolah::is_blocking_move(uint8_t player, Move m) const {    
    uint64_t other_bb   = bitboard(other_player(player));
    uint64_t to_bb      = square_bb(m.to_sq());
    uint64_t free       = free_squares() & ~to_bb;
    auto blocked = [&](uint64_t pos) {
        uint64_t north = shift<NORTH>(pos);
        uint64_t south = shift<SOUTH>(pos);
        uint64_t east = shift<EAST>(pos);
        uint64_t west = shift<WEST>(pos);
        uint64_t north_east = shift<NORTH_EAST>(pos);
        uint64_t south_east = shift<SOUTH_EAST>(pos);
        uint64_t north_west = shift<NORTH_WEST>(pos);
        uint64_t south_west = shift<SOUTH_WEST>(pos);
        return ((north | south | east | west | north_east | south_east | north_west | south_west) & free) == 0;
    };
    uint64_t north      = shift<NORTH>(to_bb);
    uint64_t south      = shift<SOUTH>(to_bb);
    uint64_t east       = shift<EAST>(to_bb);
    uint64_t west       = shift<WEST>(to_bb);
    uint64_t north_east = shift<NORTH_EAST>(to_bb);
    uint64_t south_east = shift<SOUTH_EAST>(to_bb);
    uint64_t north_west = shift<NORTH_WEST>(to_bb);
    uint64_t south_west = shift<SOUTH_WEST>(to_bb);
    return (other_bb & north) && blocked(north) ||
        (other_bb & south) && blocked(south) ||
        (other_bb & east) && blocked(east) ||
        (other_bb & west) && blocked(west) ||
        (other_bb & north_east) && blocked(north_east) ||
        (other_bb & south_east) && blocked(south_east) ||
        (other_bb & north_west) && blocked(north_west) ||
        (other_bb & south_west) && blocked(south_west);
}

bool Yolah::is_blocking_move(Move m) const {
    return is_blocking_move(current_player(), m);
}

void Yolah::moves(uint64_t bb, MoveList& moves) const {
    Move* moveList = moves.moveList;
    uint64_t occupied = black | white | empty;
    while (bb) {
        Square   from = pop_lsb(bb);
        uint64_t b    = attacks_bb(from, occupied) & ~occupied;
        while (b) {
            *moveList++ = Move(from, pop_lsb(b));
        }
    }
    if (moveList == moves.moveList) [[unlikely]] {
        *moveList++ = Move::none();
    }
    moves.last = moveList;
}

void Yolah::encircling_or_escaping_moves(uint8_t player, MoveList& moves) const {
    uint64_t player_bb = bitboard(player);
    uint64_t bb = between_n_m_liberties(player_bb, 1, 2);
    if (bb) {
        Yolah::moves(bb, moves);
        if (moves[0] == Move::none()) {
            moves.last--;
        }
    } else if (uint64_t other_bb = other_player(player); bb = between_n_m_liberties(other_bb, 1, 2)) {
        Move* moveList = moves.moveList;
        uint64_t occupied = black | white | empty;
        while (player_bb) {
            Square   from = pop_lsb(player_bb);
            uint64_t b    = attacks_bb(from, occupied);
            if (b & bb) {
                b &= ~occupied;
                while (b) {
                    *moveList++ = Move(from, pop_lsb(b));
                }
            }     
        }
        moves.last = moveList;
    }
}

uint64_t Yolah::free_squares() const {
    uint64_t occupied = black | white | empty;
    return FULL & ~occupied;
}

uint64_t Yolah::bitboard(uint8_t player) const {
    return player == BLACK ? black : white;
}

uint64_t Yolah::around(uint64_t bb) const {
    uint64_t north      = shift<NORTH>(bb);
    uint64_t south      = shift<SOUTH>(bb);
    uint64_t east       = shift<EAST>(bb);
    uint64_t west       = shift<WEST>(bb);
    uint64_t north_east = shift<NORTH_EAST>(bb);
    uint64_t south_east = shift<SOUTH_EAST>(bb);
    uint64_t north_west = shift<NORTH_WEST>(bb);
    uint64_t south_west = shift<SOUTH_WEST>(bb);
    return north | south | east | west | north_east | south_east | north_west | south_west;
}

uint64_t Yolah::between_n_m_liberties(uint64_t player_bb, size_t n, size_t m) const {
    uint64_t res = 0;
    uint64_t free = free_squares();
    while (player_bb) {
        uint64_t bb = player_bb & -player_bb;
        if (size_t k = std::popcount(around(bb) & free); n <= k && k <= m) {
            res |= bb;
        }
        player_bb &= ~bb;
    }
    return res;
}

string Yolah::to_json() const {
    json j;
    j["ply"]   = to_string(ply);
    j["black"] = to_string(black);
    j["white"] = to_string(white);
    j["empty"] = to_string(empty);
    j["black score"] = to_string(black_score);
    j["white score"] = to_string(white_score);
    return j.dump();
}

Yolah Yolah::from_json(std::istream& is) {
    json j = json::parse(is);
    Yolah res;
    res.ply   = static_cast<uint16_t>(stoi(j["ply"].get<string>()));
    res.black = stoull(j["black"].get<string>());
    res.white = stoull(j["white"].get<string>());
    res.empty = stoull(j["empty"].get<string>());
    res.black_score = static_cast<uint16_t>(stoi(j["black score"].get<string>()));
    res.white_score = static_cast<uint16_t>(stoi(j["white score"].get<string>()));
    return res;
}

Yolah Yolah::from_json(const std::string& s) {
    std::stringstream ss;
    ss << s;
    return from_json(ss);
}

ostream& operator<<(ostream& os, const Yolah& yolah) {
    char grid[8][8];
    uint64_t black = yolah.black;
    uint64_t white = yolah.white;
    uint64_t empty = yolah.empty;    
    for (int i = 0; i < 8; i++) {        
        for (int j = 0; j < 8; j++) {
           if (black & uint64_t(1) << j) grid[i][j] = 'X';
           else if (white & uint64_t(1) << j) grid[i][j] = 'O'; 
           else if (empty & uint64_t(1) << j) grid[i][j] = ' ';
           else grid[i][j] = '.';
        }
        black >>= 8;
        white >>= 8;
        empty >>= 8;
    }
    const char* letters = "    a   b   c   d   e   f   g   h";
    const char* line = "  +---+---+---+---+---+---+---+---+";
    os << letters << '\n';
    for (int i = 7; i >= 0; i--) {
        os << line << '\n';
        os << i + 1 << ' ';
        for (int j = 0; j < 8; j++) {
            os << "| " << grid[i][j] << " ";
        }
        os << "| " << i + 1 << '\n';
    }
    os << line << '\n';
    os << letters << '\n';
    const auto [black_score, white_score] = yolah.score();
    os << "score: " << black_score << '/' << white_score << '\n';
    return os;
}
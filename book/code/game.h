#ifndef GAME_H
#define GAME_H
#include <iostream>
#include <vector>
#include <utility>
#include "types.h"
#include "move.h"
#include "json.hpp"
#include "misc.h"
using json = nlohmann::json;
class Yolah {
    uint64_t black = BLACK_INITIAL_POSITION;
    uint64_t white = WHITE_INITIAL_POSITION;
    uint64_t empty = 0;
    uint16_t black_score = 0;
    uint16_t white_score = 0;
    uint16_t ply = 0;
public:
    static constexpr uint16_t MAX_NB_MOVES = 75;
    static constexpr int MAX_NB_PLIES = 120;
    class MoveList {
      Move moveList[MAX_NB_MOVES], *last;
    public:
      explicit MoveList() : last(moveList) {}
      const Move* begin() const { return moveList; }
      const Move* end() const { return last; }
      Move* begin() { return moveList; }
      Move* end() { return last; }
      size_t size() const { return last - moveList; }
      const Move& operator[](size_t i) const { return moveList[i]; }
      Move& operator[](size_t i) { return moveList[i]; }
      Move* data() { return moveList; }
      friend class Yolah;
    };
    static constexpr uint8_t BLACK = 0;
    static constexpr uint8_t WHITE = 1;
    static constexpr uint8_t EMPTY = 2;
    static constexpr uint8_t FREE  = 3;
    constexpr std::pair<uint16_t, uint16_t> score() const;
    constexpr int16_t score(uint8_t player) const;
    constexpr uint8_t current_player() const;
    static constexpr uint8_t other_player(uint8_t player);
    uint8_t get(Square) const;
    uint8_t get(File f, Rank r) const;
    bool game_over() const;
    void play(Move m);
    void undo(Move m);
    void moves(uint8_t player, MoveList& moves) const;
    void moves(MoveList& moves) const;
    bool valid(Move m) const;
    constexpr uint64_t free_squares() const;
    constexpr uint64_t occupied_squares() const;
    constexpr uint64_t bitboard(uint8_t player) const;
    constexpr uint16_t nb_plies() const;
    constexpr uint64_t empty_bitboard() const;
    std::string to_json() const;
    static Yolah from_json(std::istream& is);
    static Yolah from_json(const std::string&);
    friend std::ostream& operator<<(std::ostream& os, const Yolah& yolah);
};
std::ostream& operator<<(std::ostream& os, const Yolah& yolah);
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
#endif

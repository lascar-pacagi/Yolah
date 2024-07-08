#ifndef GAME_H
#define GAME_H

#include <iostream>
#include <vector>
#include <utility>
#include "types.h"
#include "move.h"
#include "json.hpp"

using json = nlohmann::json;

/*

    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 | O | . | . | . | . | . | . | X | 8
  +---+---+---+---+---+---+---+---+
7 | . | . | . | . | . | . | . | . | 7
  +---+---+---+---+---+---+---+---+
6 | . | . | . | . | . | . | . | . | 6
  +---+---+---+---+---+---+---+---+
5 | . | . | . | X | O | . | . | . | 5
  +---+---+---+---+---+---+---+---+
4 | . | . | . | O | X | . | . | . | 4
  +---+---+---+---+---+---+---+---+
3 | . | . | . | . | . | . | . | . | 3
  +---+---+---+---+---+---+---+---+
2 | . | . | . | . | . | . | . | . | 2
  +---+---+---+---+---+---+---+---+
1 | X | . | . | . | . | . | . | O | 1
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h

*/

class Yolah {
    uint64_t black = BLACK_INITIAL_POSITION;
    uint64_t white = WHITE_INITIAL_POSITION;
    uint64_t empty = 0;
    uint16_t black_score = 0;
    uint16_t white_score = 0;
    uint16_t ply = 0;

public:
    static constexpr uint16_t MAX_NB_MOVES = 75;
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
      Move* data() { return moveList; }
      friend class Yolah;
    };
    static constexpr uint8_t BLACK = 0;
    static constexpr uint8_t WHITE = 1;
    std::pair<uint16_t, uint16_t> score() const;
    int16_t score(uint8_t player) const;
    uint8_t current_player() const;
    static constexpr uint8_t other_player(uint8_t player) {
      return 1 - player;
    }
    bool game_over() const;
    void play(Move m);
    void undo(Move m);
    void moves(uint8_t player, MoveList& moves) const;
    void moves(MoveList& moves) const;
    bool valid(Move m) const;
    uint64_t free_squares() const;
    uint64_t bitboard(uint8_t player) const;
    std::string to_json() const;
    static Yolah from_json(std::istream& is);
    static Yolah from_json(const std::string&);
    friend std::ostream& operator<<(std::ostream& os, const Yolah& yolah);
};

std::ostream& operator<<(std::ostream& os, const Yolah& yolah);

#endif

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
    constexpr std::pair<uint16_t, uint16_t> score() const {
      return { black_score, white_score };
    }
    constexpr int16_t score(uint8_t player) const {
      return (black_score - white_score) * ((player == WHITE) * -1 + (player == BLACK)); 
    }
    constexpr uint8_t current_player() const {
      return uint8_t(ply & 1);
    }
    static constexpr uint8_t other_player(uint8_t player) {
      return 1 - player;
    }
    uint8_t get(Square) const;
    uint8_t get(File f, Rank r) const;
    bool game_over() const;
    void play(Move m);
    void undo(Move m);
    void moves(uint8_t player, MoveList& moves) const;
    void moves(MoveList& moves) const;
    void moves(uint64_t bb, MoveList& moves) const;
    void blocking_moves(uint8_t player, MoveList& moves) const;
    void blocking_moves(MoveList& moves) const;
    bool is_blocking_move(uint8_t player, Move) const;
    bool is_blocking_move(Move) const;
    // TO DO : add definition of contact move
    void contact_moves(uint8_t player, MoveList& moves) const;
    void contact_moves(MoveList& moves) const;
    bool is_contact_move(uint8_t player, Move) const;
    bool is_contact_move(Move) const;
    bool valid(Move m) const;
    constexpr uint64_t free_squares() const {
      return FULL & ~occupied_squares();
    }
    constexpr uint64_t occupied_squares() const {
      return black | white | empty;
    }
    constexpr uint64_t bitboard(uint8_t player) const {
      return player == BLACK ? black : white;
    }
    constexpr uint16_t nb_plies() const {
      return ply;
    }
    constexpr uint64_t empty_bitboard() const {
      return empty;
    }
    constexpr bool contact(Move m) const {
      return around(square_bb(m.to_sq())) & (black | white) & ~square_bb(m.from_sq());
    }
    std::string to_json() const;
    static Yolah from_json(std::istream& is);
    static Yolah from_json(const std::string&);
    friend std::ostream& operator<<(std::ostream& os, const Yolah& yolah);
    
    void set_state(uint64_t black, 
                    uint64_t white,
                    uint64_t empty,
                    uint16_t black_score,
                    uint16_t white_score,
                    uint16_t ply) {      
      this->black = black;
      this->white = white;
      this->empty = empty;
      this->black_score = black_score;
      this->white_score = white_score;
      this->ply = ply;
    }
};

std::ostream& operator<<(std::ostream& os, const Yolah& yolah);

#endif

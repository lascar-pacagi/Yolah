// Minimal shim for the encoder build.
//
// The real ../data/generate_games.h pulls in "player.h" → a heavy dependency
// tree we do NOT want when building just the feature encoder.
// yolah_features.cpp only uses one symbol from generate_games: `data::decode_game`.
// This shim declares it (the encoder CLI provides the definition inline) and
// nothing else.
#pragma once
#include <vector>
#include <cstdint>
#include "move.h"

namespace data {
    void decode_game(uint8_t* encoding, std::vector<Move>& moves,
                     int& nb_moves, int& nb_random_moves,
                     int& black_score, int& white_score);
}

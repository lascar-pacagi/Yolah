// encode_features_cli.cpp
// =======================
// Tiny CLI wrapper around YolahFeatures::encode_data.
//
// Reads `games_*` binary game files from <src_dir>, replays each game position
// by position, calls YolahFeatures::set_features on each position, and writes
// per-source-file `<basename>.features.txt` binary records into <dst_dir>:
//
//   bytes 0 .. NB_FEATURES-1  : feature bytes (uint8)
//   byte  NB_FEATURES         : 3-class outcome label  (0 black, 1 draw, 2 white)
//
// (This is the SAME format the existing C++ pipeline writes and the
// features_net119x256x64x1 Python code already understands.)
//
// Usage:
//   encode_features <src_dir> <dst_dir>
//
// Build notes
// -----------
// yolah_features.cpp `#include "generate_games.h"` but only references
// `data::decode_game` from that header. The real generate_games.h pulls
// player.h → a heavy dependency tree, so we point -I at an `encoder_shim/`
// directory that contains a minimal generate_games.h declaring only
// data::decode_game. The definition is provided HERE so the link succeeds
// without compiling generate_games.cpp.
#include "yolah_features.h"
#include "magic.h"
#include "zobrist.h"
#include "move.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>

// The one symbol yolah_features.cpp references from generate_games.h. Defining
// it here avoids linking against generate_games.cpp (and its player.h tree).
namespace data {
    void decode_game(uint8_t* encoding, std::vector<Move>& moves, int& nb_moves,
                     int& nb_random_moves, int& black_score, int& white_score) {
        nb_moves        = encoding[0];
        nb_random_moves = encoding[1];
        for (int i = 0, k = 2; i < nb_moves; i++, k += 2) {
            moves[i] = Move(Square(encoding[k]), Square(encoding[k + 1]));
        }
        black_score = encoding[nb_moves * 2 + 2];
        white_score = encoding[nb_moves * 2 + 3];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <src_dir> <dst_dir>\n"
                  << "  src_dir: directory containing games_* binary files\n"
                  << "  dst_dir: directory to write *.features.txt files\n";
        return 1;
    }
    magic::init();
    zobrist::init();
    YolahFeatures::encode_data(argv[1], argv[2]);
    return 0;
}

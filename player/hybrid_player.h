#ifndef HYBRID_PLAYER_H
#define HYBRID_PLAYER_H
// hybrid_player.h — two players, one after the other: "early" plays the moves
// of the first plies of the game, "late" plays all the others.
//
// The idea (user, 2026-11-22): at the start of the game a big network (MCTS
// ResNet on the GPU) may judge better than a fast but shallow minimax, which
// only reaches small depths there (many free squares, huge branching); later
// the minimax NNUE searches deep, and from 22 free squares on its endgame
// solver plays exactly. The switch ply is measured by campaign (config/hybrid/).
//
// JSON configuration:
// {
//   "name": "HybridPlayer",
//   "switch ply": 18,           // early plays while yolah.nb_plies() < 18, late from ply 18 on
//   "microseconds": 1000000,    // optional: time per move, copied into both players
//   "nb threads": 4,            // optional: copied into the late player only (the
//                               // early one, a GPU MCTS, keeps its threads: its batches)
//   "early": { ...a player config... },
//   "late":  { ...a player config... }
// }
// The plies count from the initial position, the opening's plies included: with
// the 6-ply balanced book, "switch ply": 18 gives the early player 6 moves of
// each side (plies 6 to 17).
//
// Each player keeps its own state between its moves (the MCTS its tree, the
// minimax its tables). The late player starts with empty tables at the switch:
// its first move costs it a little, as for the first move of a game.
#include "player.h"
#include <memory>

class HybridPlayer : public Player {
    std::unique_ptr<Player> early, late;
    uint16_t switch_ply;
    json cfg;
public:
    explicit HybridPlayer(const json& j);
    Move play(Yolah yolah) override;
    std::string info() override;
    void game_over(Yolah yolah) override;
    json config() override { return cfg; }
};

#endif

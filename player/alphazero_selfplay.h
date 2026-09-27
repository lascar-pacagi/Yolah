#ifndef ALPHAZERO_SELFPLAY_H
#define ALPHAZERO_SELFPLAY_H
// alphazero_selfplay.h — the self-play half of the AlphaZero learning loop.
//
// The loop (driven by nnue/alphazero_learn.py, see player/README_ALPHAZERO_LEARN.md):
//
//   self-play (this file, C++) ──samples──▶ <work>/selfplay/*.bin
//        ▲                                          │
//        │ latest.json                              ▼
//   <work>/models/az_<step>.ts ◀──export── trainer (alphazero_learn.py)
//
// run_selfplay() plays many games at once, one az::Search per game, all the
// searches sharing ONE network behind an nn::BatchingEvaluator: with ~100
// games each asking for a few leaves at a time, the GPU sees large batches
// while every individual search stays close to sequential (few virtual-loss
// collisions). It polls <work>/latest.json and hot-swaps the weights as soon
// as the trainer exports a new network.
//
// Only the "full" searches of playout cap randomization are recorded (KataGo:
// the fast searches exist to make games longer/cheaper, their visit counts are
// too noisy to be a policy target). A sample is the position, the policy
// target π over its legal moves and — once the game is over — its outcome z.
//
// Optional auxiliary targets (KataGo's ownership / score heads, see the doc,
// "KataGo's auxiliary heads"): when the trainer runs with --aux it writes
// "aux": true in latest.json, and the games then record, for every sample,
// its FUTURE OWNERSHIP map — for each square, who will leave it between this
// position and the end of the game (every move leaves a hole on the square
// it comes from and scores one point, so the map sums to the rest of the
// score). Those rows are TrainingSampleAux in files of version 2; without
// the option nothing changes (TrainingSample, version 1).
#include "alphazero_mcts.h"
#include "nn_evaluator.h"
#include <atomic>
#include <cstdint>
#include <string>

namespace az {

// ── On-disk format (little-endian). Mirrored by SAMPLE_DTYPE in
//    nnue/alphazero_learn.py — change both together. ─────────────────────────
struct TrainingSample {
    uint64_t black, white, empty;        // bitboards (the network input, see nn::encode_planes)
    uint8_t  turn;                       // 0 = black to move, 1 = white
    int8_t   z;                          // outcome for the side to move: -1, 0, +1
    uint8_t  nb_moves;                   // entries used in action[] / prob[]
    uint8_t  flags;                      // reserved (0)
    float    root_q;                     // search value of the root, side to move
    uint32_t model_step;                 // trainer step of the network that searched
    uint16_t action[Yolah::MAX_NB_MOVES];  // policy index from*64+to (nn::action_index)
    uint16_t prob[Yolah::MAX_NB_MOVES];    // π(action) · 65535, rounded
};
static_assert(sizeof(TrainingSample) == 336, "keep in sync with alphazero_learn.py");

// Future ownership of a square, relative to the side to move at the sample.
enum OwnershipCode : uint8_t {
    OWN_NONE = 0,   // nobody will leave it (stays free, or a piece ends the game there)
    OWN_MINE = 1,   // the side to move will leave it: +1 for the side to move
    OWN_OPP  = 2,   // the opponent will leave it: +1 for the opponent
    OWN_PAST = 3,   // already a hole: who made it is not in the position → no target
};

// A version-2 row: the version-1 row followed by the future ownership map,
// 2 bits per square, square q in byte q/4 at bits 2·(q%4). Mirrored by
// SAMPLE_DTYPE_AUX in nnue/alphazero_learn.py.
struct TrainingSampleAux {
    TrainingSample base;
    uint8_t        own[16];
};
static_assert(sizeof(TrainingSampleAux) == 352, "keep in sync with alphazero_learn.py");

struct SampleFileHeader {
    char     magic[8];                   // "YOLAHSP1"
    uint32_t version;                    // 1 = TrainingSample, 2 = TrainingSampleAux
    uint32_t sample_size;                // sizeof(TrainingSample)
    uint32_t nb_samples;
    uint32_t nb_games;
};
static_assert(sizeof(SampleFileHeader) == 24);

// The network self-play must use, published by the trainer in
// <work>/latest.json as {"step": 12000, "ts": "models/az_00012000.ts"}, plus
// "aux": true when the trainer wants the auxiliary targets. A relative path
// is relative to the work directory.
struct ModelRef {
    uint64_t    step = 0;
    std::string path;
    bool        aux = false;                 // record TrainingSampleAux rows
};
bool read_latest_model(const std::string& work_dir, ModelRef& out);

struct SelfPlayOptions {
    json        player;                  // AlphaZeroMCTSPlayer keys: search + backend (weights ignored)
    std::string work_dir;                // reads latest.json, writes selfplay/
    std::string tag;                     // part of the file names (default host_pid)
    size_t      nb_games       = 128;    // games played concurrently
    uint64_t    max_games      = 0;      // stop after this many games (0 = until stopped)
    size_t      flush_samples  = 2000;   // write a file every N samples...
    double      flush_seconds  = 600;    // ...or every N seconds, whichever comes first
    double      poll_seconds   = 30;     // how often latest.json is checked
    double      report_seconds = 300;    // how often the throughput line is printed
    // Stop when latest.json has not changed for this long (0 = never). For
    // self-play jobs running apart from the trainer (alphazero_selfplay.sh):
    // when the trainer's job is over, no new network comes and they stop
    // instead of burning GPU hours on an outdated one.
    double      max_stale_hours = 0;
};

// Plays until `stop` is set (or max_games). Unfinished games are dropped;
// finished ones are always written before returning.
void run_selfplay(const SelfPlayOptions& options, const std::atomic<bool>& stop);

} // namespace az

#endif

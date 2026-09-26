#ifndef NN_EVALUATOR_H
#define NN_EVALUATOR_H
// nn_evaluator.h — backend-agnostic interface between the AlphaZero-style MCTS
// (player/alphazero_mcts.h) and the two-headed value/policy network.
//
// The search never touches tensors: it hands the evaluator a batch of
// positions together with their legal moves and receives, per position,
//   • a scalar value v ∈ (-1, +1) from the point of view of the player to move
//     (the network's tanh output, trained on z ∈ {-1, 0, +1});
//   • the raw policy logits of *exactly the legal moves*, in the order they
//     were given. Softmax (with an optional temperature) is done by the search.
//
// Two backends implement this interface:
//   nnue/resnet.h        — pure C++ (Eigen + OpenMP), no external dependency,
//                          loads the .bin written by nnue/resnet_export.py;
//   nnue/resnet_torch.h  — libtorch/TorchScript, GPU capable, built only with
//                          -DENABLE_TORCH=ON, loads the .ts from the same script.
// plus a wrapper:
//   nnue/batched_evaluator.h — merges the requests of several search threads
//                          into one large batch before calling a backend.
//
// Learning loop (nnue/alphazero_learn.py): a freshly trained network is
// exported as TorchScript and swapped in with reload() while the self-play
// games keep running — no search object needs to be rebuilt.
#include "game.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace nn {

// ── Board encoding (MUST match preprocess.py / encode_cnn in
//    cnn_resnet_value_policy_chunked.py) ────────────────────────────────────
//
// The network sees 4 planes of 8x8:
//   plane 0 black stones, plane 1 white stones, plane 2 "empty" (vacated)
//   squares, plane 3 the side to move (all 0 = black, all 1 = white).
// Plane cell i (row-major, i = 8*row + col with row/col in 0..7) holds bit
// (63 - i) of the bitboard, i.e. square (63 - i), since bit s IS square s
// (SQ_A1 = 0 ... SQ_H8 = 63).
//
// Where the (63 - i) comes from: preprocess.py builds a plane with
//     np.unpackbits(np.array([bb], dtype='>u8').view(np.uint8)).reshape(8, 8)
// '>u8' dumps the integer big-endian (most significant byte first) and
// unpackbits takes the most significant bit of each byte first, so the values
// come out in the order bit 63, 62, ..., 0 — element i is bit 63 - i.
//
// Geometrically, with s = 8*rank + file, that is row = 7 - rank and
// col = 7 - file: the ranks are in the usual order (rank 8 on the top row)
// but the FILES ARE MIRRORED, H on the left. Cell 0 is H8, cell 63 is A1.
// The network does not care — a fixed mirror of every input is just a
// relabelling of the board — but training and inference must agree exactly,
// so do not "fix" this: it is what the weights were trained on.
constexpr int PLANES        = 4;
constexpr int PLANE_CELLS   = 64;
constexpr int INPUT_FLOATS  = PLANES * PLANE_CELLS;
constexpr int NUM_ACTIONS   = 64 * 64;

// Fill out[0..255] with the 4 planes of `y` in NCHW order (plane-major).
inline void encode_planes(const Yolah& y, float* out) {
    const uint64_t bbs[3] = { y.bitboard(Yolah::BLACK), y.bitboard(Yolah::WHITE), y.empty_bitboard() };
    for (int c = 0; c < 3; c++) {
        const uint64_t bb = bbs[c];
        float* plane = out + c * PLANE_CELLS;
        for (int i = 0; i < PLANE_CELLS; i++) {
            plane[i] = static_cast<float>((bb >> (63 - i)) & 1);
        }
    }
    const float turn = (y.current_player() == Yolah::WHITE) ? 1.0f : 0.0f;
    float* plane = out + 3 * PLANE_CELLS;
    for (int i = 0; i < PLANE_CELLS; i++) {
        plane[i] = turn;
    }
}

// Policy head index of a move: from*64 + to (preprocess.py: pol = s1*64 + s2).
// A pass (Move::none() == a1:a1) maps to index 0 exactly like in training.
constexpr int action_index(Move m) {
    return static_cast<int>(m.from_sq()) * 64 + static_cast<int>(m.to_sq());
}

// ── Batch interface ────────────────────────────────────────────────────────
struct Request {
    const Yolah* state;     // position to evaluate
    const Move*  moves;     // its legal moves (Yolah::moves), nb_moves >= 1
    uint16_t     nb_moves;
};

struct Result {
    float value;                        // for the player to move at Request::state
    float logits[Yolah::MAX_NB_MOVES];  // policy logits of moves[0..nb_moves)
};

class Evaluator {
public:
    virtual ~Evaluator() = default;
    // Evaluate in.size() positions; out.size() must be >= in.size().
    // Thread-safe: may be called concurrently from several search threads.
    virtual void evaluate(std::span<const Request> in, std::span<Result> out) = 0;
    // The batch size the backend is most efficient at (a hint for the search).
    virtual size_t preferred_batch_size() const = 0;
    // Human readable description (backend, device, weights file).
    virtual std::string info() const = 0;
    // Swap in new weights (same architecture). Must not be called while
    // evaluate() is running.
    virtual void reload(const std::string& weights_filename) = 0;
};

// Build an evaluator from the player's JSON config:
//   "backend"    : "cpu" (default) | "torch"
//   "weights"    : path to the .bin (cpu) or .ts (torch) exported file
//   "device"     : torch only — "cuda" (default if available) | "cpu"
//   "fp16"       : torch only — run the network in half precision (default true on cuda)
//   "nb eval threads" : cpu only — OpenMP threads used inside one evaluate() (0 = all)
//   "batch size" : preferred batch size reported to the search (default backend specific)
std::unique_ptr<Evaluator> make_evaluator(const json& config);

} // namespace nn

#endif

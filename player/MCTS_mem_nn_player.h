#ifndef MCTS_MEM_NN_PLAYER_H
#define MCTS_MEM_NN_PLAYER_H
#include "MCTS_mem_player.h"
#include <memory_resource>
#include "nnue_quantized.h"

class MCTSMemNNPlayer : public Player {
    NNUE_Quantized nnue;
    const std::string nnue_parameters_filename;
    struct NNUEEvalLeaf {
        NNUE_Quantized& nnue;
        float operator()(Yolah& yolah) {
            thread_local NNUE_Quantized::Accumulator acc;
            nnue.init(yolah, acc);
            const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
            float coeff = (yolah.current_player() == Yolah::BLACK ? 1 : -1);
            return coeff * black_proba - coeff * white_proba;
        }
    };
    MCTSMemPlayer<NNUEEvalLeaf> player;

public:
    explicit MCTSMemNNPlayer(uint64_t microseconds, const std::string& nnue_parameters_filename)
        : MCTSMemNNPlayer(microseconds, nnue_parameters_filename, std::thread::hardware_concurrency()) {
    }
    explicit MCTSMemNNPlayer(uint64_t microseconds, const std::string& nnue_parameters_filename, std::size_t nb_threads)
        : nnue(nnue_parameters_filename), nnue_parameters_filename(nnue_parameters_filename), player(microseconds, nb_threads, NNUEEvalLeaf{nnue}) {
    }
    Move play(Yolah yolah) override {
        return player.play(yolah);
    }
    std::string info() override;
    json config() override;
};

#endif
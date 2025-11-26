#include "compare_models.h"
#include "nnue_quantized.h"
#include <filesystem>
#include <regex>
#include "generate_games.h"
#include <algorithm>
#include "game.h"
#include <vector>

void compare_models(const std::string& model1, const std::string& model2, const std::string& src_dir, std::ostream& os) {
    using namespace std;
    NNUE_Quantized nnue1, nnue2;
    nnue1.load(model1);
    nnue2.load(model2);
    auto acc1 = nnue1.make_accumulator();
    auto acc2 = nnue2.make_accumulator();
    double nnue1_accuracy = 0;
    double nnue2_accuracy = 0;
    size_t nb_positions = 0;
    vector<Move> moves(Yolah::MAX_NB_PLIES);
    const std::filesystem::path src(src_dir);
    std::regex re_games("^games.*", std::regex_constants::ECMAScript|std::regex_constants::multiline);
    for (auto const& dir_entry : std::filesystem::directory_iterator(src_dir)) {
        auto path = dir_entry.path();
        if (!std::regex_search(path.filename().string(), re_games)) continue;
        std::cout << path << std::endl;
        auto size = filesystem::file_size(path);
        vector<uint8_t> encoding(size);
        ifstream ifs(path, ios::binary);
        ifs.read(reinterpret_cast<char*>(encoding.data()), size);
        size_t n = 0;
        while (n < size) {
            int nb_moves;
            int nb_random_moves;            
            int black_score;
            int white_score;            
            data::decode_game(encoding.data() + n, moves, nb_moves, nb_random_moves, black_score, white_score);
            Yolah yolah;
            for (int i = 0; i < nb_random_moves; i++) {
                yolah.play(moves[i]);
            }
            auto update = [&]() {
                nnue1.init(yolah, acc1);
                nnue2.init(yolah, acc2);
                const auto [black_proba1, draw_proba1, white_proba1] = nnue1.output(acc1);
                const auto [black_proba2, draw_proba2, white_proba2] = nnue2.output(acc2);
                double max_proba1 = max({black_proba1, draw_proba1, white_proba1});
                double max_proba2 = max({black_proba2, draw_proba2, white_proba2});
                nnue1_accuracy += (max_proba1 == black_proba1 && black_score > white_score) 
                    + (max_proba1 == white_proba1 && white_score > black_score) 
                    + (max_proba1 == draw_proba1 && black_score == white_score);
                nnue2_accuracy += (max_proba2 == black_proba2 && black_score > white_score) 
                    + (max_proba2 == white_proba2 && white_score > black_score) 
                    + (max_proba2 == draw_proba2 && black_score == white_score);
                nb_positions++;
            };
            update();
            for (int i = nb_random_moves; i < nb_moves; i++) {
                yolah.play(moves[i]);
                update();
            }            
            n += 2 + 2 * nb_moves + 2;
        }
    }
    os << "# of positions           : " << nb_positions << '\n';
    os << "NNUE quantized 1 accuracy: " << nnue1_accuracy / static_cast<double>(nb_positions) << '\n';
    os << "NNUE quantized 2 accuracy: " << nnue2_accuracy / static_cast<double>(nb_positions) << '\n';
}

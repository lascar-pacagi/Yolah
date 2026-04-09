#include "ffnn.h"
#include <iostream>
#include "yolah_features.h"

// g++ -std=c++2a -O3 -march=native -mavx2 -ffast-math -funroll-loops -I../game -I../misc -I../eigen ../game/zobrist.cpp ../game/magic.cpp ../game/game.cpp nnue_quantized.cpp
int main(int argc, char* argv[]) {
    using namespace std;
    FFNN<YolahFeatures::NB_FEATURES, 128, 64, 3> net("");
    alignas(64) uint8_t features[decltype(net)::I_PADDED]{};
    auto size = filesystem::file_size(path);
    vector<uint8_t> encoding(size);
    ifstream ifs(path, ios::binary);
    ifs.read(reinterpret_cast<char *>(encoding.data()), size);
    stringbuf buffer;
    ostream bos(&buffer);
    size_t i = 0;
    while (i < encoding.size()) {
      int nb_moves = encoding[i];
      int nb_random_moves = encoding[i + 1];
      bos << nb_moves << ' ' << nb_random_moves << ' ';
      int k = i + 2;
      for (int j = 0; j < nb_moves; j++, k += 2) {
        bos << Move(Square(encoding[k]), Square(encoding[k + 1])) << ' ';
      }
      int black_score = encoding[k];
      int white_score = encoding[k + 1];
      bos << '(' << black_score << ',' << white_score << ")\n";
      i += nb_moves * 2 + 4;
    }
    os << buffer.str();

    const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
    cout << setprecision(17) << black_proba << '\n';
    cout << draw_proba << '\n';
    cout << white_proba << '\n';
}

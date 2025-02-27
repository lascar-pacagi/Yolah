#include "nnue.h"
#include <fstream>
#include <string>
#include <regex>
#include "game.h"
#include "move.h"
#include "types.h"
#include "magic.h"
#include "zobrist.h"
#include <iostream>
#include <iomanip>
#include <random>

/*
// g++ -std=c++2a -O3 -march=native -ffast-math -funroll-loops -I../game -I../misc -I../eigen ../game/zobrist.cpp ../game/magic.cpp ../game/game.cpp nnue.cpp
int main(int argc, char* argv[]) {
    using namespace std;
    NNUE<4096, 64, 64> nnue;
    nnue.load("nnue_parameters.txt");
    auto acc = nnue.make_accumulator();
    // Yolah yolah;
    // cout << yolah << '\n';
    // yolah.play(Move(make_square("a1"), make_square("a7")));
    // cout << yolah << '\n';
    // nnue.output_linear(yolah);
    //nnue.write(cout);
    ifstream ifs(argv[1], std::ifstream::in);
    regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
    size_t i = 0;
    while (ifs) {
        Yolah yolah;
        nnue.init(yolah, acc);
        string line;
        getline(ifs, line);
        if (line == "") continue;
        for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
            //nnue.init(yolah);
            const auto [black_proba, draw_proba, white_proba] = nnue.output_softmax(acc);
            cout << setprecision(17) << black_proba << '\n';
            cout << draw_proba << '\n';
            cout << white_proba << '\n';
            smatch match = *it;
            string match_str = match.str();
            //cout << match_str << '\n';
            Square sq1 = make_square(match[2].str());
            Square sq2 = make_square(match[3].str());
            //cout << sq1 << ':' << sq2 << '\n';
            Move m(sq1, sq2);
            nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);
            yolah.undo(m);
            nnue.undo(yolah.current_player(), m, acc);
            nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);
            //cout << yolah << '\n';            
        }
    }
}
*/

/*
int main() {
    magic::init();
    zobrist::init();
    NNUE<4096, 64, 64> nnue;
    auto acc = nnue.make_accumulator();
    float res = 0;
    PRNG prng(42);
    for (size_t i = 0; i < 10000; i++) {
        Yolah yolah;
        nnue.init(yolah, acc);
        Yolah::MoveList moves;
        while (!yolah.game_over()) {
            yolah.moves(moves);        
            Move m = moves[prng.rand<size_t>() % moves.size()];
            const auto [black_proba, ignore, white_proba] = nnue.output_softmax(acc);
            res += black_proba - white_proba;
            nnue.play(yolah.current_player(), m, acc);
            yolah.play(m);
        }
    }
    std::cout << res << std::endl;
}
*/
/*
using type = float;

void matmul(const type *a, const type *_b, type * __restrict__ c, size_t N) {
    alignas(32) type *b = new type[N * N];
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            b[i * N + j] = _b[j * N + i];
        }            
    }        
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[j * N + k];
            }                
        }            
    }        
}

void rinit(type* a, size_t N) {
    std::default_random_engine g(42);
    std::uniform_real_distribution<type> d;
    for (size_t i = 0; i < N; i++) {
        a[i] = d(g); 
    }
}

int main() {
    constexpr size_t N = 2048;
    alignas(32) type* a = new type[N * N];
    alignas(32) type* b = new type[N * N];
    alignas(32) type* c = new type[N * N];
    rinit(a, N * N);
    rinit(b, N * N);
    matmul(a, b, c, N);
    std::cout << c[0] << std::endl;
}
*/
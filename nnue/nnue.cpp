#include "nnue.h"
#include <fstream>
#include <string>
#include <regex>
#include "game.h"
#include "move.h"
#include "types.h"
#include <iostream>
#include <iomanip>

// g++ -std=c++2a -O3 -I../game -I../misc ../game/zobrist.cpp ../game/magic.cpp ../game/game.cpp nnue.cpp
int main(int argc, char* argv[]) {
    using namespace std;
    NNUE<4096, 64, 64, NNUE_BASIC> nnue;
    nnue.load("nnue_parameters.txt");
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
        nnue.init(yolah);
        string line;
        getline(ifs, line);
        if (line == "") continue;
        for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
            //nnue.init(yolah);
            const auto [black_proba, draw_proba, white_proba] = nnue.output_softmax();
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
            nnue.play(yolah.current_player(), m);                        
            yolah.play(m);
            //cout << yolah << '\n';            
        }
    }
}

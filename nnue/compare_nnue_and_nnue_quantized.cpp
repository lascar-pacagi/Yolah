#include "compare_nnue_and_nnue_quantized.h"
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
#include <stdlib.h>
#include <bit>
#include <algorithm>
#include "nnue.h"
#include "nnue_q6.h"

void compare_nnue_and_nnue_quantized(const std::string& filename) {
    using namespace std;
    NNUE nnue;
    nnue.load("nnue_1024x128x64x3.20.txt");
    NNUE_Q6 nnue_q;
    nnue_q.load("nnue_q_1024x128x64x3.20.txt");    
    auto acc = nnue.make_accumulator();
    auto acc_q = nnue_q.make_accumulator();
    ifstream ifs(filename, std::ifstream::in);
    regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
    size_t i = 0;
    size_t first = 0;
    size_t second = 0;
    size_t third = 0;
    size_t total = 0;
    double error = 0;
    while (ifs) {
        Yolah yolah;
 //       nnue.init(yolah, acc);
        string line;
        getline(ifs, line);
        if (line == "") continue;
        for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
            nnue.init(yolah, acc);
            nnue_q.init(yolah, acc_q);
            const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
            const auto [black_proba_q, draw_proba_q, white_proba_q] = nnue_q.output(acc_q);
            vector<pair<float, char>> res{ { black_proba, 'B' }, { draw_proba, 'D' }, { white_proba, 'W' } };
            vector<pair<float, char>> res_q{ { black_proba_q, 'B' }, { draw_proba_q, 'D' }, { white_proba_q, 'W' } };
            sort(begin(res), end(res));
            sort(begin(res_q), end(res_q));
            if (res[0].second == res_q[0].second) first++;
            if (res[1].second == res_q[1].second) second++;
            if (res[2].second == res_q[2].second) third++;
            error += abs(black_proba - black_proba_q) + abs(draw_proba - draw_proba_q) + abs(white_proba - white_proba_q);
            total++;
            smatch match = *it;
            string match_str = match.str();
            Square sq1 = make_square(match[2].str());
            Square sq2 = make_square(match[3].str());
            Move m(sq1, sq2);
            //nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);                        
        }
    }
    cout << "First : " << first / (float)total << '\n';
    cout << "Second: " << second / (float)total << '\n';
    cout << "Third : " << third / (float)total << '\n';
    cout << "Error : " << error / total << '\n';
}

int main(int argc, char* argv[]) {
    compare_nnue_and_nnue_quantized(argv[1]);
}

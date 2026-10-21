// program_search_dump_main.cpp — the positions of the program search
// (nnue/program_search.py): every STRIDE-th position of data/games, seen from
// the side to move, with what the base network says and what really happened.
//
//   ./program_search_dump GAMES_DIR BASE.quantized.txt OUT.bin [--stride 20] [--threads 16]
//
// OUT.bin, one packed record (36 bytes) per position:
//   uint64 mine, his, holes     the pieces of the side to move, the other side's, the holes
//   float  value                the base network (a NNUE_Persp file) for the side to move, in [-1, 1]
//   int8   future               final margin − current margin, for the side to move
//   int8   result               +1 / 0 / −1 for the side to move
//   uint8  free                 number of free squares
//   uint8  colour               0 black, 1 white (side to move)
//   uint32 game                 game number (the search splits train / test by game)
// The network is updated move by move as in the search (init, then play).
#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include "nnue_persp.h"
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector;
namespace fs = std::filesystem;

struct __attribute__((packed)) Rec {
    uint64_t mine, his, holes;
    float value;
    int8_t future, result;
    uint8_t free, colour;
    uint32_t game;
};
static_assert(sizeof(Rec) == 36);

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: program_search_dump GAMES_DIR BASE.quantized.txt OUT.bin [--stride S] [--threads T]\n";
        return 1;
    }
    magic::init();
    zobrist::init();
    int stride = 20, threads = 16;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (string(argv[i]) == "--stride") stride = std::stoi(argv[i + 1]);
        else if (string(argv[i]) == "--threads") threads = std::stoi(argv[i + 1]);
    }
    const NNUE_Persp net(argv[2]);
    vector<string> files;
    for (const auto& d : fs::directory_iterator(argv[1])) if (d.is_regular_file()) files.push_back(d.path());
    std::sort(files.begin(), files.end());

    std::ofstream out(argv[3], std::ios::binary);
    std::mutex mtx;
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> total{0};
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back([&] {
        NNUE_Persp::Accumulator acc;
        vector<Rec> local;
        for (size_t j; (j = next++) < files.size();) {
            std::ifstream in(files[j], std::ios::binary);
            vector<uint8_t> e((std::istreambuf_iterator<char>(in)), {});
            uint32_t game = uint32_t(j) << 20;      // file number, then game number in the file
            for (size_t i = 0; i + 1 < e.size(); game++) {
                const int n = e[i];
                if (i + 2 * n + 4 > e.size()) break;
                const int final_margin = int(e[i + 2 + 2 * n]) - int(e[i + 3 + 2 * n]);   // black − white
                Yolah y;
                net.init(y, acc);
                Yolah::MoveList moves;
                for (int k = 0; k < n; k++) {
                    const Move m(Square(e[i + 2 + 2 * k]), Square(e[i + 3 + 2 * k]));
                    y.moves(moves);
                    if (std::find(moves.begin(), moves.end(), m) == moves.end()) break;   // corrupt
                    if ((k + game) % stride == 0) {
                        const uint8_t c = y.current_player();
                        const int sign = c == Yolah::BLACK ? 1 : -1;
                        const int fut = sign * final_margin - y.score(c);
                        local.push_back({y.bitboard(c), y.bitboard(uint8_t(c ^ 1)), y.empty_bitboard(),
                                         net.value(acc, y), int8_t(std::clamp(fut, -127, 127)),
                                         int8_t(sign * final_margin > 0 ? 1 : sign * final_margin < 0 ? -1 : 0),
                                         uint8_t(std::popcount(y.free_squares())), c, game});
                    }
                    net.play(y.current_player(), m, acc);
                    y.play(m);
                }
                i += 2 * n + 4;
            }
            if (local.size() > 100000) {
                std::lock_guard lk(mtx);
                out.write((const char*)local.data(), local.size() * sizeof(Rec));
                total += local.size();
                local.clear();
            }
        }
        std::lock_guard lk(mtx);
        out.write((const char*)local.data(), local.size() * sizeof(Rec));
        total += local.size();
    });
    for (auto& t : pool) t.join();
    std::cout << total << " positions (stride " << stride << ") → " << argv[3] << "\n";
}

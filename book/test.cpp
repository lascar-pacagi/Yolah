#include <bits/stdc++.h>

using namespace std;

// constexpr uint64_t BLACK_INITIAL_POSITION =
// 0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001;
// constexpr uint64_t WHITE_INITIAL_POSITION =
// 0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000;

// class Yolah {    
//     uint64_t black = BLACK_INITIAL_POSITION;
//     uint64_t white = WHITE_INITIAL_POSITION;
//     uint64_t holes = 0;
//     uint8_t black_score = 0;    
//     uint8_t white_score = 0;    
//     uint8_t ply = 0;
// public:
// };

// int main() {
//     cout << sizeof(Yolah) << '\n';
// }

using ll = uint64_t;

// int main() {
//     random_device rd;
//     mt19937 mt(rd());
//     uniform_int_distribution<ll> d;
//     int N = 500;
//     vector<ll> keys(N);
//     for (auto& k : keys) k = d(mt);
//     cout << keys[0] << '\n';
//     map<ll, ll> seen;
//     constexpr int K = 13;
//     while (true) {
//         seen.clear();
//         ll magic = d(mt) & d(mt) & d(mt);
//         //cout << hex << magic << '\n';
//         bool found = true;
//         for (auto k : keys) {
//             ll v = k * magic >> (64 - K);
//             if (seen.count(v) && seen[v] != k) {
//                 found = false;
//                 break;
//             }
//             seen[v] = k; 
//         }
//         if (found) {
//             cout << "found magic! " << magic << " for " << K << endl;
//             break;
//         }
//     }
// }

int main() {
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<ll> d;
    unordered_map<uint64_t, int> positions;
    for (int i = 0; i < 64; i++) {
        positions[1ULL << i] = i;
    }
    constexpr int K = 7;
    auto index = [](uint64_t magic, int k, ll bitboard) {
        return bitboard * magic >> (64 - k);
    };
    while (true) {
        ll MAGIC = d(mt);
        bool found = true;
        set<uint64_t> seen;
        for (const auto [bitboard, pos] : positions) {
            int64_t i = index(MAGIC, K, bitboard);
            if (seen.contains(i)) {
                found = false;
                break;
            }
            seen.insert(i);
        }
        if (found) {
            cout << format("found magic for K = {}: {:x}\n", K, MAGIC);
            vector<int> table(1 << K);
            for (const auto [bitboard, pos] : positions) {
                table[index(MAGIC, K, bitboard)] = pos;
            }
            cout << "uint8_t bitscan[64] = {";
            for (int i = 0; i < 64; i++) {
                cout << table[i] << ',';
            }
            cout << "};\n";
            break;
        }
    }
}

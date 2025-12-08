#include <bits/stdc++.h>

using namespace std;

class PRNG {

    uint64_t s;

    uint64_t rand64() {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }

   public:
    PRNG(uint64_t seed) :
        s(seed) {
    }

    uint64_t seed() {
        return s;
    }

    template<typename T>
    T rand() {
        return T(rand64());
    }

    // Special generator used to fast init magic numbers.
    // Output values only have 1/8th of their bits set on average.
    template<typename T>
    T sparse_rand() {
        return T(rand64() & rand64() & rand64());
    }
};

int main() {
    random_device rd;
    mt19937_64 mt(rd());
    //PRNG prng(rd());
    uniform_int_distribution<uint64_t> d;
    unordered_map<uint64_t, int> positions;
    for (int i = 0; i < 64; i++) {
        positions[1ULL << i] = i;
    }
    constexpr int K = 7;
    auto index = [](uint64_t magic, int k, uint64_t bitboard) {
        return bitboard * magic >> (64 - k);
    };
    while (true) {
        uint64_t MAGIC = d(mt);
        //uint64_t MAGIC = prng.rand<uint64_t>();
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
            cout << format("found magic for K = {}: {:#x}\n", K, MAGIC);
            int size = 1 << K;
            vector<int> table(size, -1);
            for (const auto [bitboard, pos] : positions) {
                table[index(MAGIC, K, bitboard)] = pos;
            }
            cout << format("uint8_t positions[{}] = {{", size);
            for (int i = 0; i < size; i++) {
                cout << table[i] << ',';
            }
            cout << "};\n";
            break;
        }
    }
}

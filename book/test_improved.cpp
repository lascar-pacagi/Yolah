#include <bits/stdc++.h>

using namespace std;
using ll = uint64_t;

// Verify that a magic number produces no collisions
bool verify_magic(ll magic, const vector<ll>& keys, int K) {
    map<ll, ll> seen;
    for (auto k : keys) {
        ll hash_val = (k * magic) >> (64 - K);
        if (seen.count(hash_val) && seen[hash_val] != k) {
            cout << "Collision: " << hex << k << " and " << seen[hash_val]
                 << " both hash to " << hash_val << dec << endl;
            return false;
        }
        seen[hash_val] = k;
    }
    return true;
}

// Show hash distribution
void show_distribution(ll magic, const vector<ll>& keys, int K) {
    map<ll, vector<ll>> buckets;
    for (auto k : keys) {
        ll hash_val = (k * magic) >> (64 - K);
        buckets[hash_val].push_back(k);
    }

    cout << "Hash distribution: " << buckets.size() << " unique hashes for "
         << keys.size() << " keys" << endl;

    // Show collisions
    int collisions = 0;
    for (auto& [hash, ks] : buckets) {
        if (ks.size() > 1) {
            collisions++;
            cout << "  Hash " << hash << ": " << ks.size() << " keys" << endl;
        }
    }
    if (collisions == 0) {
        cout << "  No collisions - perfect hash!" << endl;
    }
}

int main() {
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<ll> d;

    constexpr int N = 500;
    constexpr int K = 13;  // 13 bits = 8192 entries

    // Generate keys
    vector<ll> keys(N);
    for (auto& k : keys) {
        k = d(mt);
    }

    cout << "Generated " << N << " random keys" << endl;
    cout << "Hash table size: " << K << " bits (" << (1 << K) << " entries)" << endl;
    cout << "First key: 0x" << hex << keys[0] << dec << endl;
    cout << endl;

    // Search for magic number
    cout << "Searching for magic number..." << endl;
    int attempts = 0;
    ll magic = 0;

    auto start = chrono::steady_clock::now();

    while (true) {
        attempts++;

        // Generate sparse magic (fewer bits set - often better for multiplication hashing)
        magic = d(mt) & d(mt) & d(mt) & d(mt);

        // Try this magic
        map<ll, ll> seen;  // FIXED: moved inside loop
        bool found = true;

        for (auto k : keys) {
            ll hash_val = (k * magic) >> (64 - K);
            if (seen.count(hash_val) && seen[hash_val] != k) {
                found = false;
                break;
            }
            seen[hash_val] = k;
        }

        if (found) {
            auto end = chrono::steady_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "✓ Found magic number after " << attempts << " attempts!" << endl;
            cout << "Time: " << duration.count() << "ms" << endl;
            cout << "Magic: 0x" << hex << magic << dec << endl;
            cout << "Popcount: " << __builtin_popcountll(magic) << " bits set" << endl;
            cout << endl;

            // Verify
            cout << "Verification:" << endl;
            if (verify_magic(magic, keys, K)) {
                cout << "✓ Magic is valid!" << endl;
            } else {
                cout << "✗ Magic is invalid (should not happen)" << endl;
            }
            cout << endl;

            // Show distribution
            show_distribution(magic, keys, K);

            break;
        }

        // Progress indicator
        if (attempts % 100000 == 0) {
            cout << "Attempts: " << attempts << "..." << endl;
        }
    }

    return 0;
}

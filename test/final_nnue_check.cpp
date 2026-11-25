// final_nnue_check — doc/final_nnue.cpp computes exactly what the dev player's
// network (NNUE_Persp, sparse product as in the dev player) computes.
//
// Random games (random moves, passes when forced) from the initial position.
// Along each game, both networks keep their accumulators move by move with
// update() and a refresh cache, as the dev player's lazy search does; the doc
// network also keeps a chain without cache. At every position, compares:
//   • the accumulators: doc chain with cache = doc chain without cache = doc
//     init from scratch = NNUE_Persp's (values of both views);
//   • value(): the same float, bit for bit;
//   • the dev player's integer value int(value · 30000).
// Built by test/final_nnue_check.sh. Prints the counts and the mismatches.
#include "final_nnue.h"
#include "nnue_persp.h"
#include "magic.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

int main(int argc, char* argv[]) {
    magic::init();
    const std::string weights = argc > 1 ? argv[1] : "models/nnue_persp_k8_them_g046_qat_ib4f_ref2u_az20_distill.quantized.txt";
    const int games = argc > 2 ? std::atoi(argv[2]) : 300;
    NNUE_Persp ref(weights);
    ref.sparse = true;                                      // the dev player's default ("sparse h1")
    final_nnue::Network doc(weights);
    auto ref_cache = std::make_unique<NNUE_Persp::RefreshCache>();
    auto doc_cache = std::make_unique<final_nnue::RefreshCache>();
    std::mt19937_64 rng(5);
    long positions = 0, bad_acc = 0, bad_value = 0, bad_int = 0, refreshes = 0;
    for (int g = 0; g < games; g++) {
        ref_cache->clear();
        doc_cache->clear();
        Yolah y;
        NNUE_Persp::Accumulator ra[2];
        auto da = std::make_unique<final_nnue::Accumulator[]>(2);   // with cache
        auto dn = std::make_unique<final_nnue::Accumulator[]>(2);   // without cache
        auto fresh = std::make_unique<final_nnue::Accumulator>();
        ref.init(y, ra[0].acc);
        doc.init(y, da[0]);
        doc.init(y, dn[0]);
        int cur = 0;
        while (true) {
            positions++;
            doc.init(y, *fresh);
            bool acc_ok = std::memcmp(da[cur].view, dn[cur].view, sizeof da[cur].view) == 0
                       && std::memcmp(da[cur].view, fresh->view, sizeof fresh->view) == 0;
            for (int v = 0; v < 2 && acc_ok; v++)
                acc_ok = std::memcmp(da[cur].view[v], ra[cur].acc + v * NNUE_Persp::H1_SIZE, sizeof da[cur].view[v]) == 0;
            if (!acc_ok && bad_acc++ < 5) std::printf("accumulator mismatch: game %d ply %d\n", g, int(y.nb_plies()));
            const float rv = ref.value(ra[cur].acc, y), dv = doc.value(da[cur], y);
            if (std::memcmp(&rv, &dv, sizeof rv) != 0 && bad_value++ < 5)
                std::printf("value mismatch: game %d ply %d: %.9g vs %.9g\n", g, int(y.nb_plies()), rv, dv);
            if (int(rv * 30000) != int(dv * 30000)) bad_int++;
            if (y.game_over()) break;
            Yolah::MoveList ms;
            y.moves(ms);
            const Move m = ms[rng() % ms.size()];
            const uint8_t player = y.current_player();
            const int holes_before = std::popcount(y.empty_bitboard());
            ref.update(ra[cur].acc, ra[1 - cur].acc, player, m, ref_cache.get());
            doc.update(da[cur], da[1 - cur], player, m, doc_cache.get());
            doc.update(dn[cur], dn[1 - cur], player, m, nullptr);
            y.play(m);
            if (std::popcount(y.empty_bitboard()) * 4 / 57 != holes_before * 4 / 57) refreshes++;
            cur = 1 - cur;
        }
    }
    std::printf("%d games, %ld positions (%ld changes of fc1 set): accumulator mismatches %ld, value mismatches %ld, "
                "integer value mismatches %ld\n", games, positions, refreshes, bad_acc, bad_value, bad_int);
    return bad_acc || bad_value || bad_int;
}

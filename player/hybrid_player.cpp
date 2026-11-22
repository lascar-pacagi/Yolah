#include "hybrid_player.h"
#include <stdexcept>

using std::invalid_argument;

HybridPlayer::HybridPlayer(const json& j) : cfg(j) {
    for (const char* k : {"switch ply", "early", "late"})
        if (!j.contains(k)) throw invalid_argument(std::string("HybridPlayer: ") + k + " key expected");
    switch_ply = j["switch ply"].get<uint16_t>();
    json e = j["early"], l = j["late"];
    // The common time control and thread count (set by the tournament on this
    // config's top level) are passed down to the two players.
    if (j.contains("microseconds")) {
        e["microseconds"] = j["microseconds"];
        l["microseconds"] = j["microseconds"];
        if (e.contains("nb simulations")) e["nb simulations"] = 0;   // time-limited search
        if (l.contains("nb simulations")) l["nb simulations"] = 0;
    }
    if (j.contains("nb threads")) l["nb threads"] = j["nb threads"];
    early = Player::create(e);
    late  = Player::create(l);
}

Move HybridPlayer::play(Yolah yolah) {
    return yolah.nb_plies() < switch_ply ? early->play(yolah) : late->play(yolah);
}

std::string HybridPlayer::info() {
    return "hybrid (switch at ply " + std::to_string(switch_ply) + "): " + early->info() + " | " + late->info();
}

void HybridPlayer::game_over(Yolah yolah) {
    early->game_over(yolah);
    late->game_over(yolah);
}

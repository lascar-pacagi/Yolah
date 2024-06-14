#include "human_player.h"
#include <iostream>
#include <sstream>
#include <regex>
#include "heuristic.h"

namespace {
    const std::regex pattern("\\[(.*)\\]\\s*(\\w\\d):(\\w\\d)\\s*");    
}

HumanPlayer::HumanPlayer(std::unique_ptr<WebsocketServerSync> connexion) : connexion(std::move(connexion)) {
}

void HumanPlayer::send_game_state(Yolah yolah) {
    std::stringbuf wbuffer;
    std::ostream os(&wbuffer);
    os << "[game state]" << yolah.to_json();
    connexion->write(net::buffer(wbuffer.str()));
    connexion->write(net::buffer(std::to_string(heuristic::eval(yolah.current_player(), yolah))));
}

Move HumanPlayer::play(Yolah yolah) {    
    send_game_state(yolah);    
    beast::flat_buffer rbuffer;
    connexion->read(rbuffer);  
    std::smatch m;
    std::string s = beast::buffers_to_string(rbuffer.data());
    std::regex_match(s, m, pattern);
    if (m.size() == 4) {
        Square from = make_square(m[2].str());
        Square to = make_square(m[3].str());
        return Move(from, to);
    }
    return Move::none();
}

void HumanPlayer::game_over(Yolah yolah) {
    send_game_state(yolah);
}

HumanPlayer::~HumanPlayer() {
    try {
        connexion->close();
    } catch (std::exception const& e) {
        std::cerr << "~HumanPlayer error: " << e.what() << std::endl;
    }
}
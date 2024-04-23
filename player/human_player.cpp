#include "human_player.h"
#include <iostream>
#include <sstream>
#include <regex>

namespace {
    const std::regex pattern("\\[(.*)\\]\\s*(\\w\\d):(\\w\\d)\\s*");
}

HumanPlayer::HumanPlayer(std::shared_ptr<WebsocketServerSync> connexion) : connexion(connexion) {
}

Move HumanPlayer::play(Yolah yolah) {    
    std::stringbuf wbuffer;
    std::ostream os(&wbuffer);
    os << "[game state]" << yolah.to_json();
    connexion->write(net::buffer(wbuffer.str()));
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

HumanPlayer::~HumanPlayer() {
    try {
        connexion->close();
    } catch (std::exception const& e) {
        std::cerr << "~HumanPlayer error: " << e.what() << std::endl;
    }
}
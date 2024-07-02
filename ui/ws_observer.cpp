#include "ws_observer.h"

WSObserver::WSObserver(const std::string& host, uint16_t port) : ws(host, port) {
}

void WSObserver::operator()(Yolah yolah) {
    std::stringbuf buffer;
    std::ostream os(&buffer);
    os << "[game state]" << yolah.to_json();
    ws.write(net::buffer(buffer.str()));
}

void WSObserver::operator()(uint8_t player, Move m) {
    // std::stringbuf buffer;
    // std::ostream os(&buffer);
    // os << "[" << (player == Yolah::BLACK ? "black" : "white") <<  " move]" << m;
    // ws.write(net::buffer(buffer.str()));
}
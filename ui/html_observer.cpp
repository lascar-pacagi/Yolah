#include "html_observer.h"

HtmlObserver::HtmlObserver(const std::string& host, uint16_t port) : ws(host, port) {
}

void HtmlObserver::operator()(Yolah yolah) {
    std::stringbuf buffer;
    std::ostream os(&buffer);
    os << "[game state]" << yolah.to_json();
    ws.write(net::buffer(buffer.str()));
}

void HtmlObserver::operator()(uint8_t player, Move m) {
    std::stringbuf buffer;
    std::ostream os(&buffer);
    os << "[" << (player == Yolah::BLACK ? "black" : "white") <<  " move]" << m;
    ws.write(net::buffer(buffer.str()));
}
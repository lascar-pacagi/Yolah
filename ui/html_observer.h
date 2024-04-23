#ifndef HTML_OBSERVER_H
#define HTML_OBSERVER_H
#include "game.h"
#include "misc.h"

class HtmlObserver {
    WebsocketServerSync ws;
public:
    HtmlObserver(const std::string& host, uint16_t port);
    void operator()(Yolah);
    void operator()(uint8_t player, Move);
};

#endif
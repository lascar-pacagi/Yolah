#ifndef WS_OBSERVER_H
#define WS_OBSERVER_H
#include "game.h"
#include "misc.h"

class WSObserver {
    WebsocketServerSync ws;
public:
    WSObserver(const std::string& host, uint16_t port);
    void operator()(Yolah);
    void operator()(uint8_t player, Move);
};

#endif

#ifndef CLIENT_PLAYER_H
#define CLIENT_PLAYER_H
#include <memory>
#include "player.h"
#include <string>
#include "misc.h"

class ClientPlayer {
    std::unique_ptr<Player> player;
    std::shared_ptr<WebsocketClientSync> client;
public:
    ClientPlayer(std::unique_ptr<Player> player, std::unique_ptr<WebsocketClientSync> client);
    void run();
};

#endif


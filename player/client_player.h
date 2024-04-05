#ifndef CLIENT_PLAYER_H
#define CLIENT_PLAYER_H
#include <memory>
#include "player.h"

class ClientPlayer {
    std::unique_ptr<Player> player;
    std::string host;
    std::string port;
public:
    ClientPlayer(std::unique_ptr<Player> player, std::string host, std::string port);
    void run();
};

#endif


#ifndef CLIENT_PLAYER_H
#define CLIENT_PLAYER_H
#include <memory>
#include "player.h"
#include <string>
#include "misc.h"
#include "json.hpp"
#include <optional>


class ClientPlayer {
    std::unique_ptr<Player> player;
    std::shared_ptr<WebsocketClientSync> client;
    nlohmann::json read();
    void write(const std::string&);
public:
    ClientPlayer(std::unique_ptr<Player> player, std::unique_ptr<WebsocketClientSync> client);
    void run(std::optional<std::string> join_key);
};

#endif


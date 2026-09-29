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
    std::string description;
    nlohmann::json read();
    void write(const std::string&);
public:
    // description: what the server shows of this player to its opponent and
    // to the observers (the "description" key of the player's configuration);
    // empty = the player's info().
    ClientPlayer(std::unique_ptr<Player> player, std::unique_ptr<WebsocketClientSync> client,
                 std::string description = "");
    void run(std::optional<std::string> join_key);
};

#endif


#ifndef HUMAN_PLAYER_H
#define HUMAN_PLAYER_H
#include "player.h"
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include "misc.h"

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

class HumanPlayer : public Player {
    std::unique_ptr<WebsocketServerSync> connexion;
    void send_game_state(Yolah);
public:
    HumanPlayer(std::unique_ptr<WebsocketServerSync>);
    Move play(Yolah) override;
    std::string info() override;
    void game_over(Yolah) override;
    ~HumanPlayer() override;
};

#endif
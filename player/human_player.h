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
    std::shared_ptr<WebsocketServerSync> connexion;
public:
    HumanPlayer(std::shared_ptr<WebsocketServerSync>);
    Move play(Yolah yolah) override;
    ~HumanPlayer() override;
};

#endif
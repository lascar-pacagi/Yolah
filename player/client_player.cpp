#include "client_player.h"
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

ClientPlayer::ClientPlayer(std::unique_ptr<Player> player, std::unique_ptr<WebsocketClientSync> client) 
    : player(std::move(player)), client(std::move(client)) {
}

void ClientPlayer::run() {
    std::stringbuf wbuffer;
    std::ostream os(&wbuffer);
    os << "[init]";
    client->write(net::buffer(wbuffer.str()));
    beast::flat_buffer rbuffer;
    client->read(rbuffer);
    std::cout << beast::buffers_to_string(rbuffer.data()) << std::endl;
}

#include "misc.h"
#include <boost/asio/connect.hpp>
#include <iostream>


SocketServerSync::SocketServerSync(const std::string& host, uint16_t port) : socket(ioc) {
    try {
        tcp::acceptor acceptor{ioc, {net::ip::make_address(host), port}};        
        acceptor.accept(socket);                
    } catch (std::exception const& e) {
        std::cerr << "SocketServerSync error: " << e.what() << std::endl;
        throw;
    }
}

WebsocketServerSync::WebsocketServerSync(const std::string& host, uint16_t port) : SocketServerSync(host, port), ws(std::move(socket)) {
    try {
        ws.accept();
    } catch (std::exception const& e) {
        std::cerr << "WebsocketServerSync error: " << e.what() << std::endl;
        throw;
    }
}

void WebsocketServerSync::close() {
    try {
        ws.close(websocket::close_code::normal);        
    } catch (std::exception const& e) {
        std::cerr << "WebsocketServerSync::close error: " << e.what() << std::endl;
        throw;
    }    
}

std::shared_ptr<WebsocketServerSync> WebsocketServerSync::create(const std::string& host, uint16_t port) {
    try {        
        return std::make_shared<WebsocketServerSync>(host, port);
    } catch (std::exception const& e) {
        std::cerr << "Connexion::create error: " << e.what() << std::endl;
        throw;
    }
}

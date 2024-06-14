#include "misc.h"
#include <boost/asio/connect.hpp>
#include <iostream>
#include <cstdlib>

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

WebsocketServerSync::~WebsocketServerSync() {
    close();
}

std::unique_ptr<WebsocketServerSync> WebsocketServerSync::create(const std::string& host, uint16_t port) {
    try {        
        return std::make_unique<WebsocketServerSync>(host, port);
    } catch (std::exception const& e) {
        std::cerr << "Connexion::create error: " << e.what() << std::endl;
        throw;
    }
}

WebsocketClientSync::WebsocketClientSync(const std::string& host, uint16_t port) : ws(ioc) {
    try {
        tcp::resolver resolver(ioc);
        auto results = resolver.resolve(host, std::to_string(port));
        auto ep = net::connect(ws.next_layer(), results);
        ws.handshake(host, "/");                
    } catch (std::exception const& e) {
        std::cerr << "WebsocketClientSync error: " << e.what() << std::endl;
        throw;
    }
}

WebsocketClientSync::~WebsocketClientSync() {
    close();
}

void WebsocketClientSync::close() {
    try {
        ws.close(websocket::close_code::normal);        
    } catch (std::exception const& e) {
        std::cerr << "WebsocketClientSync::close error: " << e.what() << std::endl;
        throw;
    }
}

std::unique_ptr<WebsocketClientSync> WebsocketClientSync::create(const std::string& host, uint16_t port) {
    try {        
        return std::make_unique<WebsocketClientSync>(host, port);
    } catch (std::exception const& e) {
        std::cerr << "Connexion::create error: " << e.what() << std::endl;
        throw;
    }
}

void* aligned_pages_alloc(size_t alloc_size) {
    constexpr size_t alignment = 4096;
    size_t size = ((alloc_size + alignment - 1) / alignment) * alignment;
    void*  mem  = std::aligned_alloc(alignment, size);
    return mem;
}
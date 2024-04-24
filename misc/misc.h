#ifndef MISC_H 
#define MISC_H 
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <cstdint>
#include <memory>
#include <optional>

// xorshift64star Pseudo-Random Number Generator
// This class is based on original code written and dedicated
// to the public domain by Sebastiano Vigna (2014).
// It has the following characteristics:
//
//  -  Outputs 64-bit numbers
//  -  Passes Dieharder and SmallCrush test batteries
//  -  Does not require warm-up, no zeroland to escape
//  -  Internal state is a single 64-bit integer
//  -  Period is 2^64 - 1
//  -  Speed: 1.60 ns/call (Core i7 @3.40GHz)
//
// For further analysis see
//   <http://vigna.di.unimi.it/ftp/papers/xorshift.pdf>

class PRNG {

    uint64_t s;

    uint64_t rand64() {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }

   public:
    PRNG(uint64_t seed) :
        s(seed) {
    }

    template<typename T>
    T rand() {
        return T(rand64());
    }

    // Special generator used to fast init magic numbers.
    // Output values only have 1/8th of their bits set on average.
    template<typename T>
    T sparse_rand() {
        return T(rand64() & rand64() & rand64());
    }
};

inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
    uint64_t aL = uint32_t(a), aH = a >> 32;
    uint64_t bL = uint32_t(b), bH = b >> 32;
    uint64_t c1 = (aL * bL) >> 32;
    uint64_t c2 = aH * bL + c1;
    uint64_t c3 = aL * bH + uint32_t(c2);
    return aH * bH + (c2 >> 32) + (c3 >> 32);
}

static constexpr bool DEBUG = false;

void debug(auto&& print) {
    if constexpr (DEBUG) {
        print();
    }
}

namespace net = boost::asio;
namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;

class SocketServerSync {
protected:
    net::io_context ioc;
    tcp::socket socket;
public:
    SocketServerSync(const std::string& host, uint16_t port);
};

class WebsocketServerSync : public SocketServerSync {    
    websocket::stream<tcp::socket> ws;
public:
    WebsocketServerSync(const std::string& host, uint16_t port);
    void read(auto&& buffer) {
        ws.read(buffer);
    }
    void write(auto&& buffer) {
        ws.write(buffer);
    }
    void close();
    static std::shared_ptr<WebsocketServerSync> create(const std::string& host, uint16_t port);
};

#endif
#include "client_player.h"
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <map>
#include <sstream>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;
using std::string, std::cout, std::endl;
using std::stringbuf, std::ostream;

namespace {
    struct Message {
        enum {
            Error,
            Init,
            New,
            Join,
            Watch,
            Chat,
            Info,
            GameState,
            YourMove,
            MyMove
        };
        static constexpr std::map<std::string, int> type2enum {
            { "error",  Error },
            { "init", Init },
            { "new", New },
            { "join", Join },
            { "watch", Watch },
            { "chat", Chat },
            { "info", Info },
            { "game state", GameState },
            { "your move", YourMove },
            { "my move", MyMove }
        };
        static int type(const json& msg) {
            return type2enum[msg["type"].get<string>()];
        }
        static string new_(const string& info) {
            json j;
            j["type"] = "new";
            j["info"] = info;
            return j.dump();
        }
        static string join_key(const json& msg) {
            return msg["join key"].get<string>();
        }
        static string watch_key(const json& msg) {
            return msg["watch key"].get<string>();
        }
        static string my_move(Move m) {
            stringbuf buffer;
            ostream os(&buffer);    
            os << m;
            json j;
            j["type"] = "my move";
            j["move"] = wbuffer.str();
            return j.dump();
        }
        static string game_state(const json& msg) {
            json j = msg["state"];
            return j.dump();
        }
        static string chat(const json& msg) {
            return msg["message"].get<string>();
        }
        static string error(const json& msg) {
            return msg["message"].get<string>();
        }
    };
}

ClientPlayer::ClientPlayer(std::unique_ptr<Player> player, std::unique_ptr<WebsocketClientSync> client) 
    : player(std::move(player)), client(std::move(client)) {
}

json ClientPlayer::read() {
    beast::flat_buffer rbuffer;
    client->read(rbuffer);
    return json::parse(rbuffer.data());
}

void ClientPlayer::write(const std::string& msg) {
    client->write(net::buffer(msg));
}

void ClientPlayer::run(std::optional<std::string> join_key) {
    Yolah yolah;
    if (!join_key) write(Message::new_(player->info()));
    else write(Message::join(*join_key));
    for (;;) {
        json msg = read();
        switch Message::type(msg) {
            case Message::Error:
                cout << Message::error(msg) << endl;
                break;
            case Message::Init:
                cout << "joint key: " << Message::join_key(msg) << endl;
                cout << "watch key: " << Message::watch_key(msg) << endl;                
                break;
            case Message::YourMove:
                write(Message::my_move(player->play(yolah)));
                break;
            case Message::GameState:
                yolah.from_json(Message::game_state(msg));
                if (yolah.game_over()) return;
                break;
            case Message::Chat:
                cout << Message::chat(msg) << endl;
                break;
        }
    }
}

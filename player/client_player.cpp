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
        static const std::map<std::string, int> type2enum;
        static int type(const json& msg) {
            return type2enum.at(msg["type"].get<string>());
        }
        static string new_(const string& info) {
            json j;
            j["type"] = "new";
            j["info"] = info;
            return j.dump();
        }
        static string join(const string& key, const string& info) {
            json j;
            j["type"]     = "join";
            j["join key"] = key;
            j["info"]     = info;
            return j.dump();
        }
        static string get_join_key(const json& msg) {
            return msg["join key"].get<string>();
        }
        static string get_watch_key(const json& msg) {
            return msg["watch key"].get<string>();
        }
        static string my_move(Move m) {
            stringbuf buffer;
            ostream os(&buffer);    
            os << m;
            json j;
            j["type"] = "my move";
            j["move"] = buffer.str();
            return j.dump();
        }
        static string get_game_state(const json& msg) {
            json j = msg["state"];
            return j.dump();
        }
        static string get_chat(const json& msg) {
            return msg["message"].get<string>();
        }
        static string get_error(const json& msg) {
            return msg["message"].get<string>();
        }
    };
    const std::map<std::string, int> Message::type2enum {
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
}

ClientPlayer::ClientPlayer(std::unique_ptr<Player> player, std::unique_ptr<WebsocketClientSync> client) 
    : player(std::move(player)), client(std::move(client)) {
}

json ClientPlayer::read() {
    //beast::flat_buffer rbuffer;
    string data;
    auto buffer = net::dynamic_buffer(data);
    client->read(buffer);
    //client->read(rbuffer);
    return json::parse(data);
}

void ClientPlayer::write(const std::string& msg) {
    client->write(net::buffer(msg));
}

void ClientPlayer::run(std::optional<string> join_key) {
    Yolah yolah;
    if (!join_key) write(Message::new_(player->info()));
    else write(Message::join(*join_key, player->info()));
    for (;;) {
        json msg = read();
        cout << msg << endl;
        switch (Message::type(msg)) {
            case Message::Error:
                cout << Message::get_error(msg) << endl;
                break;
            case Message::Init:
                cout << "join key:  " << Message::get_join_key(msg) << endl;
                cout << "watch key: " << Message::get_watch_key(msg) << endl;                
                break;
            case Message::YourMove:
                write(Message::my_move(player->play(yolah)));
                break;
            case Message::GameState:
                yolah = Yolah::from_json(Message::get_game_state(msg));
                if (yolah.game_over()) return;
                break;
            case Message::Chat:
                cout << "chat" << endl;
                cout << Message::get_chat(msg) << endl;
                break;
        }
    }
}

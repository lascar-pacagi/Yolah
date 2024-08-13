import asyncio
import json
import secrets
import websockets
from yolah import Yolah, Move
from enum import Enum, auto


# Protocol
# 
# "type": ("error" | "init" | "new" | "join" | "watch" | "chat" | "info" | "game state" | "your move" | "my move")    
# 
# { 
#   "type": "error"
#   "message": <string>
# }
# { 
#   "type": "new"
#   "info": <string>
# }
# { 
#   "type": "init"
#   "join key": <string key>
#   "watch key": <string key>
# }
# { 
#   "type": "join"
#   "join key": <string key>
#   "info": <string>
# }
# { 
#   "type": "watch"
#   "watch key": <string key>
#   "info": <string>
# }
# { 
#   "type": "chat"
#   "message": <string>
# }
# { 
#   "type": "info"
#   "command": <string>
# }
# { 
#   "type": "info"
#   "result": <string>
# }
# { 
#   "type": "game state"
#   "state": {
#       "black": <black bitboard string>
#       "white": <white bitboard string>
#       "empty": <empty bitboard string>
#       "black score": <black score string>
#       "white_score": <white score string>
#       "ply": <ply string>
#    }
# }
# { 
#   "type": "your move"
# }
# { 
#   "type": "my move"
#   "move": <move string>
# }

class Connected:
    def __init__(self):
        self.players = []
        self.observers = []
        
    def add_player(self, websocket, info):
        self.players.append((websocket, info))

    def add_observer(self, websocket, info):
        self.observers.append((websocket, info))

    def remove(self, websocket):
        self.players = list(filter(lambda x: x[0] != websocket, self.players))
        self.observers = list(filter(lambda x: x[0] != websocket, self.observers))

    def other_player(self, websocket):
        return next(map(lambda x: x[0], filter(lambda x: x[0] != websocket, self.players)))

    def others(self, websocket):
        return list(map(lambda x: x[0], filter(lambda x: x[0] != websocket, self.players + self.observers)))

    def connections(self):
        return list(map(lambda x: x[0], self.players + self.observers))    

    def nb_players(self):
        return len(self.players)

    def nb_observers(self):
        return len(self.nb_observers)

    def players_info(self):
        return ", ".join((info for (_, info) in self.players))

    def observers_info(self):
        return ", ".join((info for (_, info) in self.observers))

    def __len__(self):
        return len(self.players) + len(self.observers)


class Message(Enum):
    ERROR = auto()
    INIT = auto()
    NEW = auto()
    JOIN = auto()
    WATCH = auto() 
    CHAT = auto() 
    INFO = auto()
    GAME_STATE = auto() 
    YOUR_MOVE = auto() 
    MY_MOVE = auto()

    @staticmethod 
    def type(msg):        
        return {
            "error": Message.ERROR, "init": Message.INIT, "new": Message.NEW, "join": Message.JOIN, 
            "watch": Message.WATCH, "chat": Message.CHAT, "info": Message.INFO, "game state": Message.GAME_STATE, 
            "your move": Message.YOUR_MOVE, "my move": Message.MY_MOVE
        }[msg["type"]]
        
    @staticmethod
    def error(msg):
        return json.dumps({
            "type": "error",
            "message": msg
        })
    
    @staticmethod 
    def init(join_key, watch_key):
        return json.dumps({
            "type": "init",
            "join key": join_key,
            "watch key": watch_key
        })

    @staticmethod
    def your_move():
        return json.dumps({
            "type": "your move"
        })
    
    @staticmethod
    def game_state(state):
        return json.dumps({
            "type": "game state",
            "state": state
        })
    
    @staticmethod
    def chat(msg):
        return json.dumps({
            "type": "chat",
            "chat": msg
        })

    @staticmethod
    def get_command(msg):
        return msg["command"]
    
    @staticmethod
    def get_info(msg):
        return msg["info"]
    
    @staticmethod
    def get_join_key(msg):
        return msg["join key"]
    
    @staticmethod
    def get_watch_key(msg):
        return msg["watch key"]

    @staticmethod
    def get_move(msg):
        return Move.from_str(msg["move"])
    
    @staticmethod
    def get_chat(msg):
        return msg["message"]
    
    @staticmethod
    def info(result):
        return json.dumps({
            "type": "info",
            "result": result
        })


JOIN  = {}
WATCH = {}


async def error(websocket, msg):
    print(f'[error] {msg}')
    await websocket.send(Message.error(msg))


async def handle_message(websocket, msg, game = None, connected = None):
    match Message.type(msg):
        case Message.CHAT:
            if connected:
                others = connected.others(websocket)
                print(Message.get_chat(msg))
                if others:
                    websockets.broadcast(others, json.dumps(msg))
        case Message.INFO:
            match Message.get_command(msg):
                case "join list":
                    result = "\n"
                    if len(JOIN.items()):
                        result += "available games\n---------------" if len(JOIN) else "available game\n--------------"
                        result += "\n"
                        for k, (_, c) in JOIN.items():
                            if c.nb_players() == 1:
                                result += k + ": " + c.players_info() + "\n"
                    else:
                        result = "no game\n"
                    await websocket.send(Message.info(result))
                case "watch list":
                    result = "\n"
                    if len(WATCH.items()):
                        result += "available games to follow\n-------------------------" if len(JOIN) else "available game to follow\n-------------------------"
                        result += "\n"
                        for k, (_, c) in WATCH.items():
                            result += k + ": " + c.players_info() + "\n"
                    else:
                        result = "no game\n"
                    await websocket.send(Message.info(result))


async def play(websocket, game, connected):
    async for message in websocket:
        msg = json.loads(message)
        match Message.type(msg):
            case Message.MY_MOVE:
                m = Message.get_move(msg)
                game.play(m)
                websockets.broadcast(connected.connections(), Message.game_state(json.loads(game.to_json())))
                if game.game_over():
                    return
                else:
                    await connected.other_player(websocket).send(Message.your_move())
            case _:
                await handle_message(websocket, msg, game, connected)


async def join(websocket, msg):
    try:
        game, connected = JOIN[Message.get_join_key(msg)]
    except KeyError:
        await error(websocket, "Game not found.")
        return
    connected.add_player(websocket, Message.get_info(msg))
    try:
        await websocket.send(Message.game_state(json.loads(game.to_json())))
        await connected.other_player(websocket).send(Message.your_move())
        await play(websocket, game, connected)
    finally:
        connected.remove(websocket)


async def watch(websocket, msg):
    try:
        game, connected = WATCH[Message.get_watch_key(msg)]
    except KeyError:
        await error(websocket, "Game not found.")
        return
    connected.add_observer(websocket, Message.get_info(msg))
    try:
        await websocket.send(Message.game_state(json.loads(game.to_json())))
        async for message in websocket:
            msg = json.loads(message)
            await handle_message(websocket, msg, game, connected)
    finally:
        connected.remove(websocket)


async def start(websocket, msg):
    game = Yolah()
    connected = Connected()
    connected.add_player(websocket, Message.get_info(msg))
    join_key = secrets.token_urlsafe(12)
    JOIN[join_key] = game, connected
    watch_key = secrets.token_urlsafe(12)
    WATCH[watch_key] = game, connected
    try:
        await websocket.send(Message.init(join_key, watch_key))
        await play(websocket, game, connected)
    finally:
        del JOIN[join_key]
        del WATCH[watch_key]


async def connect(websocket):
    print("[connexion]")
    while True:
        message = await websocket.recv()
        msg = json.loads(message)
        print(msg)
        match Message.type(msg):
            case Message.NEW: 
                await start(websocket, msg)
                break
            case Message.JOIN:
                await join(websocket, msg)
                break
            case Message.WATCH:
                await watch(websocket, msg)
                break
            case _:
                await handle_message(websocket, msg)


async def main():
    async with websockets.serve(connect, "", 8001, ping_timeout=120):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

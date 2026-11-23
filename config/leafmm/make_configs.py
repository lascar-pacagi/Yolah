#!/usr/bin/env python3
"""make_configs.py — the players of campaign H2 (nnue/leafmm_campaign.sh).

The MCTS ResNet of the app (config/hybrid/resnet.cfg) whose late leaves are
valued by a short minimax search (dev_u9_spsa, config/hybrid/dev.cfg), see
player/leaf_minimax.h. One config per (min ply P, nodes N):

    config/leafmm/mm_P_N.cfg       the player
    config/leafmm/players_P_N.txt  the 2-player tournament: it against dev

    python3 config/leafmm/make_configs.py      (from the repository root)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HYB = os.path.join(HERE, "..", "hybrid")
PLIES = [int(x) for x in os.environ.get("PLIES", "24 32 40").split()]
NODES = [int(x) for x in os.environ.get("NODES", "1000 4000 16000").split()]

resnet = json.load(open(os.path.join(HYB, "resnet.cfg")))
dev = json.load(open(os.path.join(HYB, "dev.cfg")))
leaf_player = {k: v for k, v in dev.items() if k not in ("name", "description", "nb threads", "microseconds", "tt size", "verbose")}

for p in PLIES:
    for n in NODES:
        name = f"mm_{p}_{n}"
        c = dict(resnet)
        c["description"] = (f"MCTS ResNet (app) whose leaves from ply {p} on are valued by a {n}-node dev_u9_spsa search "
                            f"(4 at a time: the opponent's 4 threads)")
        c["leaf minimax"] = {"min ply": p, "nodes": n, "microseconds": 50000, "max depth": 64, "weight": 1.0,
                             "concurrency": 4, "tt size": 16, "player": leaf_player}
        json.dump(c, open(os.path.join(HERE, name + ".cfg"), "w"), indent=4)
        with open(os.path.join(HERE, f"players_{p}_{n}.txt"), "w") as f:
            f.write(f"# campaign H2: {name} against dev_u9_spsa (nnue/leafmm_campaign.sh)\n{name}.cfg\n../hybrid/dev.cfg\n")
        print(name)

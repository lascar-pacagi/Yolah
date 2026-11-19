"""
spsa.py — SPSA tuning of the dev player's numeric parameters, as Stockfish's
Fishtest does (Spall's simultaneous perturbation, Fishtest's schedules).

At each batch, ALL the parameters are moved at once by a random ±c_k (one
random sign per parameter, Δ); θ+ = θ + c_k·Δ plays θ− = θ − c_k·Δ, both
colours, on balanced openings; then every parameter moves towards the side
that won:
    θ_i ← θ_i + R_k · c_k,i · (W − L) · Δ_i         (W, L: the wins of θ+, of θ−)
with k = the number of game PAIRS played so far, N = the pairs of the run,
    c_k,i = c_i / (k + 1)^γ,   c_i = c_end,i · N^γ                 γ = 0.101
    R_k,i = a_k,i / c_k,i²,    a_k,i = a_i / (A + k + 1)^α,  a_i = r_end · c_end,i² · (A + N)^α,  A = 0.1 N, α = 0.602
(the step at the end is r_end · c_end per unit of W − L). A batch is one call
of yolah_tournament with ROUNDS rounds (2 games each, one opening per round).

Integers: θ ± c is rounded stochastically (unbiased), so a depth with c = 0.6
is sometimes ±1, sometimes the same. Bounds: min/max of the parameter file.

Resumable: STATE (the θ, k, and the pending batch with its Δ and its seed)
is written before a batch is played; a restarted run replays the same batch,
whose results file the tournament resumes.

Files in --work:
    state.json            θ, k, pending batch
    log.csv               one line per batch: k, W, D, L, then every θ
    batch_<n>/            plus.cfg, minus.cfg, players.txt, games.csv
    current.cfg           the base config with the current θ (what the run recommends)

    python3 spsa.py --params /Yolah/config/spsa/dev_u8_params.json --work /work/spsa_dev_u8 \
        --tournament /work/build/.../yolah_tournament --pairs 50000 --rounds 64 --parallel 64 \
        --time 200000 --book /Yolah/config/openings/balanced.txt
    python3 spsa.py --report --work ...     (where every parameter went, from log.csv)
"""
import argparse
import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

GAMMA, ALPHA = 0.101, 0.602
R_END = 0.002

stop = False


def on_signal(signum, frame):
    global stop
    stop = True
    print(f"[{time.strftime('%F %T')}] signal {signum}: stopping after this batch", flush=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def schedules(p, k, n_pairs):
    A = 0.1 * n_pairs
    c = p["c_end"] * n_pairs ** GAMMA
    a = R_END * p["c_end"] ** 2 * (A + n_pairs) ** ALPHA
    ck = c / (k + 1) ** GAMMA
    ak = a / (A + k + 1) ** ALPHA
    return ck, ak / ck ** 2


def concrete(p, x, rng):
    x = min(p["max"], max(p["min"], x))
    if p["int"]:
        lo = math.floor(x)
        return int(lo + (1 if rng.random() < x - lo else 0))
    return round(x, 4)


def write_cfg(base, values, path, description):
    cfg = dict(base)
    cfg.update(values)
    cfg["description"] = description
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


def results_of(csv_path):
    """W, D, L of 'plus' against 'minus' in a tournament results file."""
    w = d = l = 0
    if not os.path.exists(csv_path):
        return 0, 0, 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            r = float(row["result"])               # 1 black wins, 0.5 draw, 0 white wins
            plus_black = row["black"] == "plus"
            s = r if plus_black else 1 - r
            if s == 1:
                w += 1
            elif s == 0:
                l += 1
            else:
                d += 1
    return w, d, l


def report(args):
    spec = load_json(args.params) if args.params else None
    log = os.path.join(args.work, "log.csv")
    if not os.path.exists(log):
        parent = os.path.dirname(os.path.normpath(args.work))
        runs = sorted(d for d in os.listdir(parent) if os.path.exists(os.path.join(parent, d, "log.csv"))) if os.path.isdir(parent) else []
        print(f"no {log} (run not started, or another NAME/PARAMS: PARAMS=<name>_params.json bash nnue/spsa.sh report)"
              + (f"; runs with a log: {', '.join(runs)}" if runs else ""))
        return
    rows = list(csv.DictReader(open(log)))
    if not rows:
        print("no batch yet")
        return
    W = sum(int(r["W"]) for r in rows); D = sum(int(r["D"]) for r in rows); L = sum(int(r["L"]) for r in rows)
    last = rows[-1]
    print(f"{len(rows)} batches, {int(last['k']):,} pairs ({2 * int(last['k']):,} games); θ+ vs θ−: W {W} D {D} L {L}")
    names = [c for c in rows[0] if c not in ("k", "W", "D", "L", "time")]
    starts = {p["name"]: p["start"] for p in spec["params"]} if spec else {}
    print(f"{'parameter':22} {'start':>9} {'now':>9} {'change':>8}   trend over the last quarter")
    q = rows[-max(1, len(rows) // 4)]
    for n in names:
        now, before = float(last[n]), float(q[n])
        st = starts.get(n, float(rows[0][n]))
        print(f"{n:22} {st:>9.4g} {now:>9.4g} {100 * (now - st) / (abs(st) or 1):>+7.1f}%   {now - before:+.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params")
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tournament")
    ap.add_argument("--config-dir", default="/Yolah/config", help="cwd of the tournament (weights ../models/...)")
    ap.add_argument("--pairs", type=int, default=50000, help="N: game pairs of the whole run")
    ap.add_argument("--rounds", type=int, default=64, help="rounds per batch (2 games each)")
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--time", type=int, default=200000)
    ap.add_argument("--book", default="")
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0: no limit)")
    args = ap.parse_args()
    if args.report:
        return report(args)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    spec = load_json(args.params)
    base = load_json(os.path.join(args.config_dir, spec["base"]))
    params = spec["params"]
    os.makedirs(args.work, exist_ok=True)
    state_path = os.path.join(args.work, "state.json")
    log_path = os.path.join(args.work, "log.csv")
    if os.path.exists(state_path):
        st = load_json(state_path)
    else:
        st = {"k": 0, "batch": 0, "theta": {p["name"]: float(p["start"]) for p in params}, "pending": None,
              "pairs": args.pairs}
        save_json(state_path, st)
    n_pairs = st["pairs"]
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(",".join(["k", "W", "D", "L", "time"] + [p["name"] for p in params]) + "\n")
    t_end = time.time() + args.hours * 3600 if args.hours > 0 else None

    while st["k"] < n_pairs and not stop and (t_end is None or time.time() < t_end):
        if st["pending"] is None:
            rng = random.Random(1_000_003 * st["batch"] + 17)
            delta = {p["name"]: rng.choice((-1, 1)) for p in params}
            plus, minus = {}, {}
            for p in params:
                ck, _ = schedules(p, st["k"], n_pairs)
                th = st["theta"][p["name"]]
                plus[p["name"]] = concrete(p, th + ck * delta[p["name"]], rng)
                minus[p["name"]] = concrete(p, th - ck * delta[p["name"]], rng)
            st["pending"] = {"delta": delta, "plus": plus, "minus": minus, "seed": 1000 + st["batch"]}
            save_json(state_path, st)
        pend = st["pending"]
        bdir = os.path.join(args.work, f"batch_{st['batch']:05d}")
        os.makedirs(bdir, exist_ok=True)
        write_cfg(base, pend["plus"], os.path.join(bdir, "plus.cfg"), f"SPSA batch {st['batch']}: θ + c·Δ")
        write_cfg(base, pend["minus"], os.path.join(bdir, "minus.cfg"), f"SPSA batch {st['batch']}: θ − c·Δ")
        with open(os.path.join(bdir, "players.txt"), "w") as f:
            f.write(os.path.join(bdir, "plus.cfg") + "\n" + os.path.join(bdir, "minus.cfg") + "\n")
        games = os.path.join(bdir, "games.csv")
        cmd = [args.tournament, "--players", os.path.join(bdir, "players.txt"), "--results", games,
               "--time", str(args.time), "--threads", "1", "--parallel", str(args.parallel),
               "--rounds", str(args.rounds), "--opening-plies", "4", "--seed", str(pend["seed"])]
        if args.book:
            cmd += ["--book", args.book]
        proc = subprocess.Popen(cmd, cwd=args.config_dir, stdout=subprocess.DEVNULL, stderr=open(os.path.join(bdir, "tournament.log"), "a"))
        while proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if stop:
                    proc.send_signal(signal.SIGTERM)
        w, d, l = results_of(games)
        if w + d + l < 2 * args.rounds:
            if stop:
                break
            print(f"batch {st['batch']}: only {w + d + l} games of {2 * args.rounds} (see {bdir}/tournament.log)", flush=True)
            time.sleep(10)
            continue
        pairs = (w + d + l) // 2
        for p in params:
            ck, rk = schedules(p, st["k"], n_pairs)
            n = p["name"]
            st["theta"][n] = min(p["max"], max(p["min"], st["theta"][n] + rk * ck * (w - l) * pend["delta"][n]))
        st["k"] += pairs
        st["batch"] += 1
        st["pending"] = None
        save_json(state_path, st)
        with open(log_path, "a") as f:
            f.write(",".join([str(st["k"]), str(w), str(d), str(l), time.strftime("%F %T")]
                             + [f"{st['theta'][p['name']]:.5g}" for p in params]) + "\n")
        cur = {p["name"]: (int(round(st["theta"][p["name"]])) if p["int"] else round(st["theta"][p["name"]], 4)) for p in params}
        write_cfg(base, cur, os.path.join(args.work, "current.cfg"),
                  f"SPSA of {spec['base']} after {st['k']} pairs ({2 * st['k']} games)")
        if st["batch"] % 10 == 0:
            print(f"[{time.strftime('%F %T')}] batch {st['batch']}, {st['k']:,}/{n_pairs:,} pairs; last W {w} D {d} L {l}", flush=True)
    print(f"[{time.strftime('%F %T')}] stopped at {st['k']:,}/{n_pairs:,} pairs", flush=True)


if __name__ == "__main__":
    main()

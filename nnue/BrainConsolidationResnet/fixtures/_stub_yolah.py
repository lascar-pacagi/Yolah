"""Side-effect-only module: installs a STUB `yolah` in sys.modules.

Imported by every test that loads an exercise that does
`from yolah import Yolah, Move, Square`. The real ../server/yolah.py is not
always present — and even when it is, tests should not depend on the actual
Yolah game logic. We give the test environment just enough.
"""
import sys
import types


def _install():
    if "yolah" in sys.modules:
        return

    mod = types.ModuleType("yolah")

    class Square:
        """Stub: holds an integer index 0..63."""
        def __init__(self, idx):
            self.idx = int(idx)
        def __repr__(self):
            return f"Square({self.idx})"

    class Move:
        """Stub: (from_sq, to_sq)."""
        def __init__(self, from_sq, to_sq):
            self.from_sq = from_sq
            self.to_sq   = to_sq

    class Yolah:
        """Stub: minimal API used by encode_cnn()."""
        def __init__(self):
            self.black = 0
            self.white = 0
            self.empty = 0
            self._ply  = 0
        def nb_plies(self):
            return self._ply
        def play(self, move):
            self._ply += 1

    mod.Square = Square
    mod.Move   = Move
    mod.Yolah  = Yolah
    sys.modules["yolah"] = mod


_install()

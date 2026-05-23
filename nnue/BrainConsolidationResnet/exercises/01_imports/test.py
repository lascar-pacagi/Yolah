"""Test for exercise 01 — imports & module setup."""
import sys
from pathlib import Path

# Make the framework's helpers and the stub yolah importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Inject a stub `yolah` module so `from yolah import Yolah, Move, Square`
# succeeds without the real ../server tree present.
sys.path.insert(0, str(ROOT / "fixtures"))
import _stub_yolah  # noqa: F401  (side-effect: registers `yolah` in sys.modules)

from _test_utils import load_exercise


REQUIRED_NAMES = [
    "tqdm", "torch", "nn", "mp", "DDP",
    "init_process_group", "destroy_process_group",
    "os", "sys", "gc", "json", "random", "threading", "queue_mod", "np",
    "Yolah", "Move", "Square",
]


def test_required_names_bound():
    ex = load_exercise(__file__)
    missing = [n for n in REQUIRED_NAMES if not hasattr(ex, n)]
    assert not missing, f"missing imports: {missing}"


def test_tf32_set_high():
    """Module-level setting: TF32 enabled for matmul precision."""
    import torch
    load_exercise(__file__)
    # The setter is global; we just check the actual value got applied.
    assert torch.get_float32_matmul_precision() == "high", (
        "expected torch.set_float32_matmul_precision('high') to be called")


def test_sharing_strategy_filesystem():
    """Module-level setting: file_system IPC sharing strategy."""
    import torch
    load_exercise(__file__)
    assert torch.multiprocessing.get_sharing_strategy() == "file_system", (
        "expected torch.multiprocessing.set_sharing_strategy('file_system')")


def test_ddp_is_distributed_data_parallel():
    """`DDP` is exactly the torch.nn.parallel.DistributedDataParallel class."""
    from torch.nn.parallel import DistributedDataParallel
    ex = load_exercise(__file__)
    assert ex.DDP is DistributedDataParallel, (
        "DDP should be torch.nn.parallel.DistributedDataParallel")


def test_queue_alias_is_module():
    """`queue_mod` must be the stdlib `queue` module, not Queue class etc."""
    import queue
    ex = load_exercise(__file__)
    assert ex.queue_mod is queue, "queue_mod must alias the `queue` module"

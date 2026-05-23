#!/usr/bin/env python3
"""runner.py — `rustlings`-style driver for BrainConsolidationResnet.

Usage:
    python3 runner.py                show all exercises + run the current one
    python3 runner.py --hint         print notes.md for the current exercise
    python3 runner.py --only 03_     run just one exercise (substring match)
    python3 runner.py --watch        re-run on file save (poll-based)

States per exercise:
    TODO   '# I AM NOT DONE' marker still present → not attempted
    PASS   marker removed, pytest green
    FAIL   marker removed, pytest red
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXERCISES_DIR = HERE / "exercises"
FIXTURES_DIR  = HERE / "fixtures"
CACHE_DIR     = FIXTURES_DIR / "cache"

NOT_DONE_MARKER = "# I AM NOT DONE"

# ANSI colours. Honoured if stdout is a TTY; piped output stays clean.
_TTY = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _TTY else s
def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def cyan(s):   return _c("36", s)
def bold(s):   return _c("1",  s)


def list_exercises():
    """Ordered list of exercise dirs matching NN_<name>."""
    if not EXERCISES_DIR.exists():
        return []
    return sorted(
        d for d in EXERCISES_DIR.iterdir()
        if d.is_dir() and re.match(r"^\d{2}_", d.name)
    )


def is_not_done(exercise_dir):
    """True if the file still has '# I AM NOT DONE' (or the file is missing)."""
    ex_py = exercise_dir / "exercise.py"
    if not ex_py.exists():
        return True
    return NOT_DONE_MARKER in ex_py.read_text()


def run_test(exercise_dir):
    """Run pytest on this exercise's test.py. Returns (passed: bool, output: str)."""
    test_py = exercise_dir / "test.py"
    if not test_py.exists():
        return False, "[no test.py present]"
    # -x stops at first failure (so we don't drown in errors).
    # --tb=short keeps the traceback compact.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q", str(test_py)],
        capture_output=True, text=True, cwd=HERE,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def ensure_fixtures():
    """Generate the synthetic cache once. Idempotent and fast."""
    if (CACHE_DIR / "meta.json").exists():
        return
    print(cyan("Building fixture cache (one-time)..."))
    proc = subprocess.run(
        [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
        cwd=HERE,
    )
    if proc.returncode != 0:
        print(red("Fixture build failed."))
        sys.exit(1)


def status_line(ex, status):
    name = ex.name
    if status == "PASS":  return f"  [{green('PASS')}]  {name}"
    if status == "FAIL":  return f"  [{red('FAIL')}]   {name}"
    if status == "TODO":  return f"  [{yellow('TODO')}]   {name}"
    return f"  [{status}]  {name}"


def print_all(exercises, current_name=None):
    print()
    for ex in exercises:
        if is_not_done(ex):
            status = "TODO"
        else:
            passed, _ = run_test(ex)
            status = "PASS" if passed else "FAIL"
        marker = " " + bold(cyan("← current")) if ex.name == current_name else ""
        print(status_line(ex, status) + marker)
    print()


def find_current(exercises):
    """The first not-done OR failing exercise. None if all pass."""
    for ex in exercises:
        if is_not_done(ex):
            return ex
        passed, _ = run_test(ex)
        if not passed:
            return ex
    return None


def cmd_default(exercises):
    current = find_current(exercises)
    print_all(exercises, current_name=current.name if current else None)

    if current is None:
        print(green(bold("All exercises complete! 🎉")))
        return 0

    if is_not_done(current):
        print(cyan(f"Current exercise: {current.name}"))
        print(f"  edit:  {current / 'exercise.py'}")
        print(f"  notes: {current / 'notes.md'}")
        print(f"  When ready, remove the `{NOT_DONE_MARKER}` line and re-run.")
        return 0

    # Marker removed but tests fail — show the output to help diagnose.
    passed, output = run_test(current)
    if not passed:
        print(red(bold(f"--- {current.name}: test output ---")))
        print(output.rstrip())
        return 1
    return 0


def cmd_hint(exercises):
    """Print notes.md for the first not-done OR failing exercise."""
    current = find_current(exercises)
    if current is None:
        print(green("All exercises complete — no hint needed."))
        return 0
    notes = current / "notes.md"
    if not notes.exists():
        print(yellow(f"No notes.md for {current.name}"))
        return 1
    print(notes.read_text())
    return 0


def cmd_only(exercises, query):
    matches = [e for e in exercises if query in e.name]
    if not matches:
        print(red(f"No exercise matches '{query}'"))
        return 1
    for ex in matches:
        if is_not_done(ex):
            print(status_line(ex, "TODO"))
            print(yellow(f"  (still has `{NOT_DONE_MARKER}` — remove it to test)"))
            continue
        passed, output = run_test(ex)
        print(status_line(ex, "PASS" if passed else "FAIL"))
        if not passed:
            print(output.rstrip())
    return 0


def cmd_watch(exercises):
    """Poll exercise.py mtimes; re-run when any changes."""
    print(cyan("Watch mode — Ctrl-C to stop."))
    last_state = {}
    try:
        while True:
            changed = False
            for ex in exercises:
                ex_py = ex / "exercise.py"
                if not ex_py.exists():
                    continue
                mtime = ex_py.stat().st_mtime
                if last_state.get(ex_py) != mtime:
                    last_state[ex_py] = mtime
                    changed = True
            if changed:
                print("\033[2J\033[H", end="")  # clear screen
                cmd_default(exercises)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hint",  action="store_true",
                        help="print notes.md for the current exercise")
    parser.add_argument("--only",  metavar="QUERY",
                        help="run exercises whose dir name contains QUERY")
    parser.add_argument("--watch", action="store_true",
                        help="re-run on file change (poll)")
    args = parser.parse_args()

    ensure_fixtures()
    exercises = list_exercises()
    if not exercises:
        print(red("No exercises found under exercises/"))
        return 1

    if args.hint:
        return cmd_hint(exercises)
    if args.only:
        return cmd_only(exercises, args.only)
    if args.watch:
        return cmd_watch(exercises)
    return cmd_default(exercises)


if __name__ == "__main__":
    sys.exit(main())

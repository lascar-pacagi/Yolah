#!/usr/bin/env python3
"""
build_app_doc.py: doc/src/yolah_app/*.html -> doc/yolah_app.html

    python3 doc/build_app_doc.py

The source is plain HTML, in parts (concatenated in the order of their file
names), with two kinds of code blocks, coloured here with
Pygments (the page needs no script and no network):

    <pre data-lang="cpp">            raw code, written as is (no HTML escaping)
    ...
    </pre>

    <pre data-extract="void fan(" data-file="app/yolah_app.cpp"></pre>
        the definition that starts on the first line containing the text, up to
        the closing brace at its indentation, copied from the source file, so
        the page always shows the current code. Optional: data-lang (default
        cpp), data-max="40" (cut after 40 lines, with a note), data-nth="2"
        (the second line containing the text).

Every block gets a caption with the file and the line numbers when extracted.
"""
import html
import os
import re
import sys

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(HERE, "src", "yolah_app")
OUT = os.path.join(HERE, "yolah_app.html")

_files = {}


def source_lines(rel):
    if rel not in _files:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            _files[rel] = f.read().split("\n")
    return _files[rel]


def extract(rel, pattern, nth=1, max_lines=0):
    lines = source_lines(rel)
    hits = [i for i, l in enumerate(lines) if pattern in l]
    if len(hits) < nth:
        sys.exit(f"build_app_doc: '{pattern}' not found in {rel}")
    start = hits[nth - 1]
    indent = len(lines[start]) - len(lines[start].lstrip())
    # A one-line definition, or a declaration ending with ';' on its first line.
    first = re.sub(r"\s*//.*$", "", lines[start]).rstrip()   # without a trailing comment
    end = start
    if not (first.endswith("{") or first.endswith("(") or first.endswith(",")):
        end = start          # one line (a declaration, a macro, a one-line function)
    else:
        for j in range(start + 1, len(lines)):
            l = lines[j]
            if l.strip() in ("}", "};", "});") and len(l) - len(l.lstrip()) == indent:
                end = j
                break
        else:
            sys.exit(f"build_app_doc: no end for '{pattern}' in {rel}")
    # Comments right above the definition belong to it.
    top = start
    while top > 0 and lines[top - 1].strip().startswith("//") and len(lines[top - 1]) - len(lines[top - 1].lstrip()) == indent:
        top -= 1
    block = [l[indent:] if l[:indent].strip() == "" else l for l in lines[top:end + 1]]
    cut = False
    if max_lines and len(block) > max_lines:
        block = block[:max_lines]
        cut = True
    return "\n".join(block), top + 1, end + 1, cut


FORMATTER = HtmlFormatter(nowrap=True)


def colour(code, lang):
    lexer = get_lexer_by_name(lang, stripnl=False)
    return highlight(code, lexer, FORMATTER).rstrip("\n")


def attrs_of(tag):
    return dict(re.findall(r'([\w-]+)="([^"]*)"', tag))


def build():
    # The source: the parts of doc/src/yolah_app/ in the order of their names.
    parts = sorted(p for p in os.listdir(SRC_DIR) if p.endswith(".html"))
    src = ""
    for p in parts:
        with open(os.path.join(SRC_DIR, p), encoding="utf-8") as f:
            src += f.read() + "\n"

    def repl(m):
        a = attrs_of(m.group(1))
        body = m.group(2)
        lang = a.get("data-lang", "cpp")
        caption = ""
        if "data-extract" in a:
            rel = a.get("data-file", "app/yolah_app.cpp")
            code, l0, l1, cut = extract(rel, html.unescape(a["data-extract"]), int(a.get("data-nth", "1")), int(a.get("data-max", "0")))
            caption = f'<span class="where">{html.escape(rel)} · lines {l0}–{l1}{" (beginning)" if cut else ""}</span>'
            if cut:
                code += "\n    // … (the rest in the file)"
        else:
            code = body.strip("\n")
            if "data-title" in a:
                caption = f'<span class="where">{html.escape(a["data-title"])}</span>'
        cls = "code" + (" small" if a.get("data-size") == "small" else "")
        return f'<figure class="{cls}">{caption}<pre><code class="hl lang-{lang}">{colour(code, lang)}</code></pre></figure>'

    out = re.sub(r"<pre(\s[^>]*data-(?:lang|extract)[^>]*)>(.*?)</pre>", repl, src, flags=re.S)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{OUT}: {len(out) // 1024} KB")


if __name__ == "__main__":
    build()

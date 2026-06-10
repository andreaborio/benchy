#!/usr/bin/env python3
"""make_dist.py — build dist/dashboard_dist.py, a single-file dashboard for people who
want the old self-contained artifact: dashboard.py with dashboard.html re-inlined as the
PAGE raw-string literal (so the dist file needs no dashboard.html next to it). Drop it
next to benchy_common.py & friends and run it like dashboard.py. Stdlib only.

The page's third-party assets (Chart.js, marked, DOMPurify, highlight.js — see NOTICE)
are NOT inlined: they are plain files served from static/vendor/ next to the script, so
this also copies static/vendor/ into dist/. Wherever you drop dashboard_dist.py, keep a
static/vendor/ directory beside it (the dist build ships one) or the page loads chartless.

  python3 make_dist.py
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN = "# >>> PAGE loader (make_dist.py replaces this block with the inlined literal) >>>\n"
END = "# <<< PAGE loader <<<\n"

def main():
    src = open(os.path.join(HERE, "dashboard.py"), encoding="utf-8").read()
    html = open(os.path.join(HERE, "dashboard.html"), encoding="utf-8").read()
    if BEGIN not in src or END not in src:
        sys.exit("make_dist: PAGE loader markers not found in dashboard.py — nothing to replace")
    if '"""' in html or html.endswith(("\\", '"')):
        sys.exit('make_dist: dashboard.html cannot be re-inlined as r"""…""" (contains a triple '
                 'quote or ends with a backslash/quote)')
    pre, rest = src.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    out = pre + 'PAGE = r"""' + html + '"""\n' + post
    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    dist = os.path.join(HERE, "dist", "dashboard_dist.py")
    with open(dist, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    vend_src = os.path.join(HERE, "static", "vendor")
    vend_dst = os.path.join(HERE, "dist", "static", "vendor")
    if os.path.isdir(vend_src):   # vendored page assets travel with the dist file
        shutil.rmtree(vend_dst, ignore_errors=True)
        shutil.copytree(vend_src, vend_dst)
    print("wrote %s (%d chars; PAGE literal: %d chars)" % (dist, len(out), len(html)))

if __name__ == "__main__":
    main()

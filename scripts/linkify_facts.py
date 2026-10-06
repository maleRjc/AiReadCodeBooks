#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiReadCodeBooks - Convert [FACT:path:lines] into Clickable GitHub Markdown Links
Ensures that reading directly on GitHub (github.com/blob/...) allows users to click
directly into the official upstream repository at the exact pinned commit and line range (#Lstart-Lend).
"""

import os
import sys
import json
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(BASE_DIR, "books")):
    # Try looking in AiReadCodeBooks directory
    alt_base = r"G:\AiReadCode\AiReadCodeBooks"
    if os.path.exists(alt_base):
        BASE_DIR = alt_base

BOOKS_DIR = os.path.join(BASE_DIR, "books")

def linkify_all_books():
    print(f"[*] Starting FACT linkification in: {BOOKS_DIR}")
    if not os.path.exists(BOOKS_DIR):
        print(f"[-] Books directory not found: {BOOKS_DIR}")
        return

    total_converted = 0
    total_files = 0

    for slug in sorted(os.listdir(BOOKS_DIR)):
        book_dir = os.path.join(BOOKS_DIR, slug)
        if not os.path.isdir(book_dir):
            continue

        meta_p = os.path.join(book_dir, "meta.json")
        if not os.path.exists(meta_p):
            continue

        with open(meta_p, "r", encoding="utf-8") as mf:
            meta = json.load(mf)

        repo = meta.get("repo", "")
        commit = meta.get("commit", meta.get("branch", "main"))

        if not repo:
            continue

        def make_fact_replacer(r, c):
            def _replace(m):
                f_path = m.group(1).strip()
                f_lines = (m.group(2) or "").strip()
                lines_disp = f":{f_lines}" if f_lines else ""
                if "-" in f_lines:
                    p = f_lines.split("-", 1)
                    anchor = f"#L{p[0]}-L{p[1]}"
                elif f_lines:
                    anchor = f"#L{f_lines}"
                else:
                    anchor = ""
                url = f"https://github.com/{r}/blob/{c}/{f_path}{anchor}"
                return f"[FACT:{f_path}{lines_disp}]({url})"
            return _replace

        replacer = make_fact_replacer(repo, commit)

        book_converted = 0
        for sub in ["zh", "en"]:
            sub_dir = os.path.join(book_dir, sub)
            if not os.path.exists(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(sub_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Find unlinked FACT tags
                matches = re.findall(r'\[FACT:([^:\]]+)(?::([^\]]+))?\](?!\()', content)
                if matches:
                    new_content = re.sub(r'\[FACT:([^:\]]+)(?::([^\]]+))?\](?!\()', replacer, content)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    book_converted += len(matches)
                    total_converted += len(matches)
                    total_files += 1

        print(f"[+] Book {slug} ({repo} @{commit[:8]}): linkified {book_converted} FACT tags")

    print(f"\n[DONE] Successfully converted {total_converted} FACT tags across {total_files} markdown files.")

if __name__ == "__main__":
    linkify_all_books()

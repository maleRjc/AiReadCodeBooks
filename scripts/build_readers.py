#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiReadCodeBooks - Standalone Multi-book Bilingual Reader Compiler
Compiles Markdown chapters (zh) and genuine translated chapters (en) into self-contained HTML readers with live code inspector,
extracts offline FACT snippets, and provides chapter-synchronized language switching.
Zero External Server Dependencies · Commit Pinned CDN + Snippets Dual-Engine
"""

import os
import sys
import json
import re
import html
import shutil
from datetime import datetime
from bs4 import BeautifulSoup

# Configure UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(ROOT_DIR, "books")):
    alt_dir = r"G:\AiReadCode\AiReadCodeBooks"
    if os.path.exists(os.path.join(alt_dir, "books")):
        ROOT_DIR = alt_dir

BOOKS_DIR = os.path.join(ROOT_DIR, "books")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

WEBSITE_EN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "website", "books", "en")
if not os.path.exists(WEBSITE_EN_DIR):
    alt_website = r"g:\AiReadCode\GitHub\AiReadCode\website\books\en"
    if os.path.exists(alt_website):
        WEBSITE_EN_DIR = alt_website

LOCAL_CODE_DIRS = {
    "vue3": r"G:\AiReadCode\Code\core",
    "tokio": r"G:\AiReadCode\Code\tokio",
    "vllm": r"G:\AiReadCode\Code\vllm",
    "nccl": r"G:\AiReadCode\Code\nccl"
}

def clean_repeated_prefix(title, lang="en"):
    """Remove redundant prefixes like 'Chapter 1: Chapter 1:' or '第 1 章：第 1 章：'."""
    if not title:
        return ""
    title = title.strip()
    if lang == "en":
        m = re.match(r'^(?:Chapter\s*0?(\d+)[:：]\s*)+(.*)$', title, re.IGNORECASE)
        if m:
            ch_num = int(m.group(1))
            rest = m.group(2).strip()
            rest = re.sub(r'^(?:Chapter\s*0?\d+[:：]\s*)+', '', rest, flags=re.IGNORECASE).strip()
            return f"Chapter {ch_num}: {rest}"
    else:
        m = re.match(r'^(?:第\s*0?(\d+)\s*章[：:]\s*)+(.*)$', title)
        if m:
            ch_num = int(m.group(1))
            rest = m.group(2).strip()
            rest = re.sub(r'^(?:第\s*0?\d+\s*章[：:]\s*)+', '', rest).strip()
            return f"第 {ch_num} 章：{rest}"
    return title

def format_inline(text, repo="", commit="", lang="zh"):
    """Format inline elements: FACT pills, code spans, bold, and escape raw HTML."""
    facts = []
    tip_title = "在右侧代码面板查看真实源码切片" if lang == "zh" else "Inspect real source slice in right code pane"

    def save_fact(m):
        f_path = m.group(1).strip()
        f_lines = (m.group(2) or "").strip()
        lines_disp = f":{f_lines}" if f_lines else ""
        safe_path = f_path.replace("'", "\\'")
        safe_lines = f_lines.replace("'", "\\'")

        pill = (
            f'<span class="fact-pill" data-file="{html.escape(f_path)}" data-lines="{html.escape(f_lines)}" '
            f'onclick="window.highlightFact(\'{safe_path}\', \'{safe_lines}\')" title="{tip_title}">'
            f'📎 <code>{html.escape(f_path)}{lines_disp}</code>'
            f'</span>'
        )
        facts.append(pill)
        return f"@@FACT_{len(facts)-1}@@"

    text = re.sub(r'\[FACT:([^:\]]+)(?::([^\]]+))?\](?:\([^)]+\))?', save_fact, text)

    codes = []
    def save_code(m):
        c = html.escape(m.group(1))
        codes.append(f'<code>{c}</code>')
        return f"@@CODE_{len(codes)-1}@@"

    text = re.sub(r'`([^`]+)`', save_code, text)

    # Escape HTML special chars in remaining text
    text = html.escape(text)

    # Convert bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Restore inline code spans
    for idx, c in enumerate(codes):
        text = text.replace(f"@@CODE_{idx}@@", c)

    # Restore fact pills
    for idx, f in enumerate(facts):
        text = text.replace(f"@@FACT_{idx}@@", f)

    return text

def parse_markdown(md_text, repo="", commit="", lang="zh"):
    """Robust Markdown to HTML parser for AiReadCode chapters."""
    _fmt = lambda t: format_inline(t, repo, commit, lang)
    copy_label = "复制" if lang == "zh" else "Copy"
    tag_label = "〔设计推断与架构权衡〕" if lang == "zh" else "[Design Inference & Architecture Tradeoffs]"

    lines = md_text.splitlines()
    html_out = []
    in_code = False
    code_lang = ""
    code_lines = []
    in_list = False
    in_table = False
    table_lines = []

    def flush_list():
        nonlocal in_list
        if in_list:
            html_out.append("</ul>")
            in_list = False

    def flush_table():
        nonlocal in_table, table_lines
        if in_table and table_lines:
            tbl = ['<div class="table-container"><table class="reader-table">']
            if len(table_lines) >= 2:
                headers = [_fmt(c.strip()) for c in table_lines[0].strip("|").split("|")]
                tbl.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
                tbl.append("<tbody>")
                for row in table_lines[2:]:
                    cols = [_fmt(c.strip()) for c in row.strip("|").split("|")]
                    tbl.append("<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>")
                tbl.append("</tbody>")
            tbl.append("</table></div>")
            html_out.append("\n".join(tbl))
            table_lines = []
            in_table = False

    for line in lines:
        s = line.strip()

        # Code fence
        if s.startswith("```"):
            flush_list()
            flush_table()
            if in_code:
                code_content = html.escape("\n".join(code_lines))
                html_out.append(
                    f'<div class="code-block-wrapper">'
                    f'<div class="code-block-header"><span>{code_lang or "code"}</span>'
                    f'<button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.closest(\'.code-block-wrapper\').querySelector(\'code\').innerText)">{copy_label}</button></div>'
                    f'<pre class="source-code-block"><code class="language-{code_lang}">{code_content}</code></pre></div>'
                )
                in_code = False
                code_lines = []
            else:
                in_code = True
                code_lang = s[3:].strip()
            continue

        if in_code:
            code_lines.append(line)
            continue

        # Details & Summary support
        if s.startswith("<details><summary>"):
            flush_list()
            flush_table()
            inner = s[len("<details><summary>"):].strip()
            if inner.endswith("</summary>"):
                inner = inner[:-len("</summary>")].strip()
            html_out.append(f'<details class="reader-details"><summary class="reader-summary">{_fmt(inner)}</summary>')
            continue
        elif s == "<details>":
            flush_list()
            flush_table()
            html_out.append('<details class="reader-details">')
            continue
        elif s.startswith("<summary>") and s.endswith("</summary>"):
            flush_list()
            flush_table()
            inner = s[len("<summary>"):-len("</summary>")].strip()
            html_out.append(f'<summary class="reader-summary">{_fmt(inner)}</summary>')
            continue
        elif s == "</details>":
            flush_list()
            flush_table()
            html_out.append('</details>')
            continue

        # Tables
        if s.startswith("|") and s.endswith("|"):
            flush_list()
            in_table = True
            table_lines.append(s)
            continue
        elif in_table:
            flush_table()

        # Lists
        if s.startswith("- ") or s.startswith("* "):
            flush_table()
            if not in_list:
                html_out.append('<ul class="reader-list">')
                in_list = True
            item_text = s[2:].strip()
            html_out.append(f'<li>{_fmt(item_text)}</li>')
            continue
        elif in_list and not s:
            flush_list()
            continue

        if not s:
            continue

        # Headers
        if s.startswith("#### "):
            flush_list()
            flush_table()
            html_out.append(f'<h4>{_fmt(s[5:])}</h4>')
        elif s.startswith("### "):
            flush_list()
            flush_table()
            html_out.append(f'<h3>{_fmt(s[4:])}</h3>')
        elif s.startswith("## "):
            flush_list()
            flush_table()
            html_out.append(f'<h2>{_fmt(s[3:])}</h2>')
        elif s.startswith("# "):
            flush_list()
            flush_table()
            html_out.append(f'<h2>{_fmt(s[2:])}</h2>')
        elif "[INFERENCE]" in s or "〔推断〕" in s:
            flush_list()
            flush_table()
            c = s.replace("[INFERENCE]", "").replace("〔推断〕", "").strip()
            html_out.append(f'<div class="inference-box"><span class="inference-tag">{tag_label}</span><p>{_fmt(c)}</p></div>')
        elif s.startswith("> "):
            flush_list()
            flush_table()
            quote_text = s[2:].strip()
            html_out.append(f'<blockquote class="reader-quote">{_fmt(quote_text)}</blockquote>')
        else:
            flush_list()
            flush_table()
            html_out.append(f'<p>{_fmt(s)}</p>')

    flush_list()
    flush_table()
    return "\n".join(html_out)

def clean_html_to_markdown(ch_html, ch_title=""):
    """Convert chapter HTML into standard clean Markdown."""
    start_idx = ch_html.find('<div class="chapter-body">')
    nav_idx = ch_html.find('<div class="chapter-nav')

    if start_idx != -1:
        if nav_idx != -1:
            content_html = ch_html[start_idx + len('<div class="chapter-body">'):nav_idx].strip()
        else:
            content_html = ch_html[start_idx + len('<div class="chapter-body">'):].strip()
        if content_html.endswith('</div>'):
            content_html = content_html[:-len('</div>')].strip()
    else:
        content_html = ch_html

    # Replace fact pills with [FACT:file:lines]
    def replace_fact(m):
        f = m.group(1)
        l = m.group(2)
        if l:
            return f"[FACT:{f}:{l}]"
        return f"[FACT:{f}]"

    content = re.sub(r'<span class="fact-pill"[^>]*data-file="([^"]+)"(?:\s+data-lines="([^"]*)")?[^>]*>.*?</span>', replace_fact, content_html, flags=re.DOTALL)

    # Replace code blocks
    def replace_code_block(m):
        lang = m.group(1) or ""
        code = m.group(2)
        code_unescaped = html.unescape(code)
        return f"\n```{lang}\n{code_unescaped}\n```\n"

    content = re.sub(r'<div class="code-block-wrapper">.*?<code class="language-([^"]*)">(.*?)</code>.*?</pre></div>', replace_code_block, content, flags=re.DOTALL)

    # Replace inference box
    def replace_inference(m):
        inner = m.group(1).strip()
        inner = re.sub(r'</?p>', '', inner)
        return f"\n> **[Design Inference & Architectural Trade-offs]**\n> {inner}\n"
    content = re.sub(r'<div class="inference-box">\s*<span class="inference-tag">.*?</span>\s*<p>(.*?)</p>\s*</div>', replace_inference, content, flags=re.DOTALL)

    # Replace blockquotes
    def replace_quote(m):
        q = m.group(1).strip()
        return f"\n> {q}\n"
    content = re.sub(r'<blockquote class="reader-quote">(.*?)</blockquote>', replace_quote, content, flags=re.DOTALL)

    # Headers
    content = re.sub(r'<h4>(.*?)</h4>', r'\n### \1\n', content)
    content = re.sub(r'<h3>(.*?)</h3>', r'\n## \1\n', content)
    content = re.sub(r'<h2>(.*?)</h2>', r'\n# \1\n', content)

    # Lists
    def replace_list(m):
        items = re.findall(r'<li>(.*?)</li>', m.group(1), re.DOTALL)
        out = []
        for it in items:
            out.append(f"- {it.strip()}")
        return "\n" + "\n".join(out) + "\n"
    content = re.sub(r'<ul class="reader-list">(.*?)</ul>', replace_list, content, flags=re.DOTALL)

    # Tables
    def replace_table(m):
        tbl = m.group(1)
        headers = re.findall(r'<th>(.*?)</th>', tbl, re.DOTALL)
        rows = re.findall(r'<tr>(.*?)</tr>', tbl, re.DOTALL)
        if not headers and not rows:
            return ""
        out = []
        if headers:
            h_line = "| " + " | ".join(h.strip() for h in headers) + " |"
            sep_line = "| " + " | ".join("---" for _ in headers) + " |"
            out.extend([h_line, sep_line])
        for r in rows:
            tds = re.findall(r'<td>(.*?)</td>', r, re.DOTALL)
            if tds:
                out.append("| " + " | ".join(td.strip() for td in tds) + " |")
        return "\n" + "\n".join(out) + "\n"
    content = re.sub(r'<div class="table-container">\s*<table class="reader-table">(.*?)</table>\s*</div>', replace_table, content, flags=re.DOTALL)

    # Inline tags: <strong>, <code>
    content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', content)
    content = re.sub(r'<code>(.*?)</code>', r'`\1`', content)

    # Paragraphs
    content = re.sub(r'<p>(.*?)</p>', r'\n\1\n', content, flags=re.DOTALL)

    # Clean any remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    content = html.unescape(content)
    content = re.sub(r'\n{3,}', '\n\n', content)

    content = content.strip()
    first_line = content.split('\n')[0].strip() if content else ""
    if first_line.startswith('#'):
        first_heading = re.sub(r'^#+\s*', '', first_line).strip()
        if ch_title and (first_heading.lower() == ch_title.lower() or 'chapter' in first_heading.lower()):
            return f"{content}\n"

    header = f"# {ch_title}\n\n" if ch_title else ""
    return f"{header}{content}\n"

def extract_book_snippets(chapters_dir, code_dir):
    """Scan all markdown files in chapters_dir, extract code lines from code_dir for all FACT tags."""
    if not os.path.exists(chapters_dir):
        return {}

    ch_files = sorted([f for f in os.listdir(chapters_dir) if f.endswith(".md")])
    unique_facts = set()
    for ch_file in ch_files:
        with open(os.path.join(chapters_dir, ch_file), "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        for m in re.finditer(r'\[FACT:([^:\]]+)(?::([^\]]+))?\](?:\([^)]+\))?', content):
            f_path = m.group(1).strip()
            f_lines = (m.group(2) or "").strip()
            unique_facts.add((f_path, f_lines))

    snippets = {}
    if not code_dir or not os.path.exists(code_dir):
        return snippets

    for f_path, f_lines in sorted(unique_facts):
        local_fp = os.path.join(code_dir, f_path.replace("/", os.sep))
        if not os.path.exists(local_fp):
            continue
        try:
            with open(local_fp, "r", encoding="utf-8", errors="ignore") as f:
                file_lines = f.readlines()
        except Exception:
            continue

        start = 1
        end = len(file_lines)
        if f_lines:
            if "-" in f_lines:
                p = f_lines.split("-")
                try:
                    start = int(p[0])
                    end = int(p[1])
                except:
                    pass
            else:
                try:
                    start = end = int(f_lines)
                except:
                    pass

        ctx_start = max(1, start - 4)
        ctx_end = min(len(file_lines), end + 4)

        items = []
        for ln in range(ctx_start, ctx_end + 1):
            items.append({
                "n": ln,
                "t": file_lines[ln - 1].rstrip("\r\n"),
                "h": 1 if (start <= ln <= end) else 0
            })

        key = f"{f_path}:{f_lines}"
        snippets[key] = {
            "file": f_path,
            "lines": f_lines,
            "items": items
        }

    return snippets

READER_TEMPLATE = """<!DOCTYPE html>
<html lang="@@HTML_LANG@@" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>@@PAGE_TITLE@@</title>
  <meta name="description" content="@@DESCRIPTION@@">
  <meta name="keywords" content="@@KEYWORDS@@">
  <link rel="icon" type="image/svg+xml" href="@@ASSETS_PATH@@favicon.svg">
  <link rel="stylesheet" href="@@ASSETS_PATH@@reader.css">
</head>
<body>
  <header class="site-header" id="navbar">
    <div class="header-container">
      <a href="@@HOME_PATH@@" class="logo-group">
        <div class="logo-icon" title="AiReadCodeBooks">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path d="M12 3.5C8.4 3.5 6 6 6 9.5C6 12 7.8 13.8 8.8 15H15.2C16.2 13.8 18 12 18 9.5C18 6 15.6 3.5 12 3.5Z" stroke="#00e5ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="9.5" y1="17.2" x2="14.5" y2="17.2" stroke="#00e5ff" stroke-width="1.6" stroke-linecap="round"/>
            <line x1="10.2" y1="19.2" x2="13.8" y2="19.2" stroke="#00e5ff" stroke-width="1.6" stroke-linecap="round"/>
            <path d="M11 21C11.5 21.6 12.5 21.6 13 21" stroke="#00e5ff" stroke-width="1.4" stroke-linecap="round"/>
          </svg>
        </div>
        <span class="logo-text">AiReadCodeBooks</span>
        <span class="badge-version">@@BADGE_VERSION@@</span>
      </a>

      <ul class="nav-links">
        <li><a href="@@HOME_PATH@@" class="nav-link">@@NAV_HOME_TEXT@@</a></li>
        <li><a href="https://github.com/maleRjc/AiReadCodeBooks" target="_blank" rel="noopener" class="nav-link">@@NAV_REPO_TEXT@@</a></li>
        <li><a href="https://github.com/@@REPO@@" target="_blank" rel="noopener" class="nav-link">@@NAV_UPSTREAM_TEXT@@</a></li>
      </ul>

      <div class="header-actions">
@@LANG_DROPDOWN_HTML@@
        <a href="@@SOURCE_URL@@" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">
          <span>@@NAV_SOURCE_TEXT@@</span>
        </a>
        <button class="theme-toggle-btn" type="button" aria-label="@@THEME_TIP@@">
          <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line></svg>
          <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
        </button>
      </div>
    </div>
  </header>

  <div class="reader-container">
    <!-- Left Sidebar: Table of Contents -->
    <aside class="reader-sidebar">
      <div class="reader-book-header">
        <span class="reader-book-badge">@@CHAPTER_BADGE@@</span>
        <h1 class="reader-book-title">@@BOOK_TITLE@@</h1>
        <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 6px; font-family: var(--font-mono);">
          <span>@@REPO@@</span> · <span>★ @@STARS@@</span>
        </div>
      </div>
      <nav aria-label="@@TOC_LABEL@@">
        <ul class="reader-toc-list" id="reader-toc">
@@TOC_HTML@@
        </ul>
      </nav>
    </aside>

    <!-- Center Column: Multi-chapter Reading View -->
    <main class="reader-main">
      <div class="book-breadcrumb">
        <a href="@@HOME_PATH@@">@@BREADCRUMB_ROOT@@</a>
        <span>/</span>
        <a href="index.html">@@BOOK_TITLE@@</a>
        <span>/</span>
        <span id="breadcrumb-current-chapter">@@BREADCRUMB_CH1@@</span>
      </div>

@@ALL_CHAPTERS_HTML@@

      <div style="margin-top: 60px; padding: 28px; background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 12px; text-align: center;">
        <h3 style="font-size: 18px; margin-bottom: 8px;">@@FOOTER_CTA_TITLE@@</h3>
        <p style="font-size: 14px; color: var(--text-secondary); max-width: 560px; margin: 0 auto 20px;">
          @@FOOTER_CTA_DESC@@
        </p>
        <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
          <a href="https://github.com/maleRjc/AiReadCodeBooks" target="_blank" rel="noopener" class="btn btn-primary btn-sm">@@FOOTER_CTA_STAR@@</a>
          <a href="@@HOME_PATH@@" class="btn btn-secondary btn-sm">@@FOOTER_CTA_BROWSE@@</a>
        </div>
      </div>
    </main>

    <!-- Right Column: Interactive Real Code Inspector (Full File View) -->
    <aside class="reader-code-pane" id="code-viewer">
      <div class="code-pane-header">
        <div class="code-pane-title-group">
          <div class="code-pane-title" id="code-pane-title-container" title="@@DEFAULT_FILE@@">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
            <span id="code-pane-filename">@@DEFAULT_FILE@@</span>
          </div>
          <span id="code-pane-lines" class="badge-lines">@@CODE_PANE_LINES@@</span>
        </div>
        <div class="code-pane-actions">
          <button class="btn-code-action" id="btn-copy-code" onclick="window.copyCurrentCode(this)" title="@@COPY_SNIPPET_TITLE@@">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span class="btn-text">@@COPY_SNIPPET_TEXT@@</span>
          </button>
          <button class="btn-code-action" id="btn-copy-all" onclick="window.copyFullFile(this)" title="@@COPY_ALL_TITLE@@">
            <span class="btn-text">@@COPY_ALL_TEXT@@</span>
          </button>
          <button class="btn-code-action" id="btn-locate-target" onclick="window.locateTargetLines()" title="@@LOCATE_TITLE@@">
            <span>@@LOCATE_TEXT@@</span>
          </button>
        </div>
      </div>
      <div class="code-pane-body" id="code-pane-content">
        <div class="code-loading-msg">@@LOADING_MSG@@</div>
      </div>
      <div class="code-pane-footer">
        <div class="code-pane-status">
          <span class="status-dot"></span>
          <span id="code-pane-file-stats">@@COMMIT_ANCHOR_TEXT@@</span>
        </div>
        <span id="code-pane-filepath" class="code-pane-filepath" title="@@DEFAULT_FILE@@">@@DEFAULT_FILE@@</span>
      </div>
    </aside>
  </div>

  <script>
    window.BOOK_META = {
      slug: "@@SLUG@@",
      repo: "@@REPO@@",
      branch: "@@BRANCH@@",
      commit: "@@COMMIT@@",
      version: "@@VERSION@@",
      defaultFile: "@@DEFAULT_FILE@@"
    };

    // Load pre-extracted snippets cache asynchronously
    fetch('@@SNIPPETS_URL@@')
      .then(res => res.ok ? res.json() : {})
      .then(data => { window.BOOK_SNIPPETS = data; })
      .catch(() => { window.BOOK_SNIPPETS = {}; });
  </script>
  <script src="@@ASSETS_PATH@@reader.js"></script>
</body>
</html>
"""

BOOKSHELF_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AiReadCodeBooks · 开源专著书库大厅</title>
  <meta name="description" content="AiReadCode 自动编撰的旗舰开源项目深度技术专著，真实代码行号永久锚定，双栏沉浸式交互阅读。">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
  <link rel="stylesheet" href="assets/reader.css">
</head>
<body>
  <header class="site-header">
    <div class="header-container">
      <a href="index.html" class="logo-group">
        <div class="logo-icon">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path d="M12 3.5C8.4 3.5 6 6 6 9.5C6 12 7.8 13.8 8.8 15H15.2C16.2 13.8 18 12 18 9.5C18 6 15.6 3.5 12 3.5Z" stroke="#00e5ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="9.5" y1="17.2" x2="14.5" y2="17.2" stroke="#00e5ff" stroke-width="1.6" stroke-linecap="round"/>
            <line x1="10.2" y1="19.2" x2="13.8" y2="19.2" stroke="#00e5ff" stroke-width="1.6" stroke-linecap="round"/>
            <path d="M11 21C11.5 21.6 12.5 21.6 13 21" stroke="#00e5ff" stroke-width="1.4" stroke-linecap="round"/>
          </svg>
        </div>
        <span class="logo-text">AiReadCodeBooks</span>
        <span class="badge-version">去中心化书库</span>
      </a>

      <ul class="nav-links">
        <li><a href="index.html" class="nav-link active">书库大厅</a></li>
        <li><a href="https://github.com/maleRjc/AiReadCodeBooks" target="_blank" rel="noopener" class="nav-link">GitHub 仓库</a></li>
        <li><a href="https://github.com/maleRjc/AiReadCode" target="_blank" rel="noopener" class="nav-link">AiReadCode 引擎</a></li>
      </ul>

      <div class="header-actions">
        <a href="https://github.com/maleRjc/AiReadCodeBooks" target="_blank" rel="noopener" class="btn btn-primary btn-sm">
          <span>GitHub Star ★</span>
        </a>
        <button class="theme-toggle-btn" type="button" aria-label="切换深色/浅色模式">
          <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line></svg>
          <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
        </button>
      </div>
    </div>
  </header>

  <main>
    <section class="bookshelf-hero">
      <h1 class="bookshelf-title">开源技术专著书库大厅</h1>
      <p class="bookshelf-desc">
        由 AiReadCode 扫描官方开源仓库全自动编撰，结合真实不可变 Commit 节点与 FACT 药丸行号溯源，提供纯静态、零服务依赖的极致双栏交互式在线阅读体验。
      </p>
    </section>

    <div class="bookshelf-grid">
@@CARDS_HTML@@
    </div>
  </main>

  <script src="assets/reader.js"></script>
</body>
</html>
"""

def render_chinese_edition(slug, book_dir, meta):
    """Render the Chinese edition from books/{slug}/zh/*.md."""
    zh_dir = os.path.join(book_dir, "zh")
    if not os.path.exists(zh_dir):
        return 0

    ch_files = sorted([f for f in os.listdir(zh_dir) if f.endswith(".md")])
    if not ch_files:
        return 0

    repo = meta.get("repo", "")
    commit = meta.get("commit", meta.get("branch", "main"))
    default_file = meta.get("defaultFile", "README.md")
    stars = meta.get("stars", "10k+")
    title = meta.get("titleZh", meta.get("title", slug))
    summary = meta.get("summaryZh", meta.get("summary", ""))

    page_title = f"《{title}》· 源码全景架构精读 | AiReadCodeBooks"
    html_lang = "zh-CN"
    assets_path = "../../assets/"
    home_path = "../../index.html"
    snippets_url = "./snippets.json"
    source_url = f"https://github.com/maleRjc/AiReadCodeBooks/tree/main/books/{slug}"
    badge_version = "在线专著"
    nav_home_text = "返回书库大厅"
    nav_repo_text = "GitHub 仓库"
    nav_upstream_text = f"目标开源: {repo}"
    nav_source_text = "查看 Markdown 源码"
    theme_tip = "切换深色/浅色模式"
    chapter_badge = f"{len(ch_files)} 章全集 · 深度专著"
    toc_label = "书籍目录导航"
    breadcrumb_root = "开源书库"
    breadcrumb_ch1 = "第 01 章"
    footer_cta_title = "读懂任意复杂项目，其实只需要一本好书"
    footer_cta_desc = "本书由 AiReadCode 扫描官方开源仓库全自动编撰而成，真实 Commit 行号永久锚定。"
    footer_cta_star = "Star GitHub 仓库 ★"
    footer_cta_browse = "浏览更多开源好书 →"
    code_pane_lines = "全览"
    copy_snippet_title = "复制当前高亮切片（若无高亮则复制全文）"
    copy_snippet_text = "复制选段"
    copy_all_title = "复制整个源文件全文"
    copy_all_text = "复制全文"
    locate_title = "回到高亮行所在位置"
    locate_text = "定位"
    loading_msg = "⚡ 正在载入源文件..."
    commit_anchor_text = "Commit 永久不可变锚定"
    keywords = f"{repo}, {title}, 源码解析, 架构设计, AiReadCode, GitHub Pages"

    lang_dropdown_html = f"""        <div class="lang-dropdown" id="lang-dropdown">
          <button class="lang-dropdown-btn" type="button" aria-haspopup="true" onclick="window.toggleLangMenu(event)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
            <span>🇨🇳 简体中文</span>
            <svg class="chevron-icon" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
          <div class="lang-dropdown-menu" id="lang-menu">
            <a href="#ch-01" class="lang-dropdown-item active" data-lang="zh" onclick="window.switchLanguage('zh', event)">🇨🇳 简体中文</a>
            <a href="en/index.html#ch-01" class="lang-dropdown-item" data-lang="en" onclick="window.switchLanguage('en', event)">🇺🇸 English</a>
          </div>
        </div>"""

    toc_items = []
    rendered_chapters = []

    meta_chapters = meta.get("chapters", [])
    ch_title_map = {}
    for c in meta_chapters:
        idx = c.get("index", 1)
        t = c.get("titleZh", f"第 {idx} 章")
        ch_title_map[idx] = clean_repeated_prefix(t, "zh")

    for idx, ch_file in enumerate(ch_files):
        order_str = f"{idx+1:02d}"
        default_ch_name = f"第 {idx+1} 章"
        ch_title = ch_title_map.get(idx+1, default_ch_name)
        clean_title = clean_repeated_prefix(ch_title, "zh")
        active_cls = " active" if idx == 0 else ""
        active_sec_cls = " active-chapter" if idx == 0 else ""

        toc_items.append(
            f'          <li><a href="#ch-{order_str}" class="toc-link{active_cls}" data-target="ch-{order_str}">'
            f'<span class="toc-num">{order_str}</span><span>{clean_title}</span></a></li>'
        )

        ch_path = os.path.join(zh_dir, ch_file)
        with open(ch_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_md = f.read()

        ch_html = parse_markdown(raw_md, repo=repo, commit=commit, lang="zh")

        prev_btn = f'<a href="#ch-{idx:02d}" class="btn btn-secondary btn-sm btn-ch-nav" data-target="ch-{idx:02d}">← 上一章：第 {idx} 章</a>' if idx > 0 else '<span></span>'
        next_btn = f'<a href="#ch-{idx+2:02d}" class="btn btn-primary btn-sm btn-ch-nav" data-target="ch-{idx+2:02d}">下一章：第 {idx+2} 章 →</a>' if idx < len(ch_files) - 1 else '<span></span>'

        commit_short = commit[:8] if commit else ""
        sec_html = f'''
      <section class="reader-chapter-section{active_sec_cls}" id="ch-{order_str}" data-title="{clean_title}">
        <div class="chapter-header">
          <span class="chapter-badge">CHAPTER {order_str}</span>
          <h2 class="reader-chapter-title">{clean_title}</h2>
          <div class="reader-chapter-meta">
            <span>官方源: {repo}</span>
            <span>·</span>
            <span>Commit @{commit_short}</span>
            <span>·</span>
            <span>全书进度: 第 {idx+1} / {len(ch_files)} 章</span>
          </div>
        </div>
        <div class="chapter-body">
          {ch_html}
        </div>
        <div class="chapter-nav-bar">
          {prev_btn}
          <a href="#reader-toc" class="btn btn-secondary btn-sm" onclick="window.scrollTo(0, 0); return false;">返回顶部 ↑</a>
          {next_btn}
        </div>
      </section>
        '''
        rendered_chapters.append(sec_html)

    toc_html = "\n".join(toc_items)
    all_chapters_html = "\n".join(rendered_chapters)
    description = summary[:160].replace('"', '&quot;')

    rendered_page = READER_TEMPLATE
    replacements = {
        "@@HTML_LANG@@": html_lang,
        "@@PAGE_TITLE@@": page_title,
        "@@BOOK_TITLE@@": title,
        "@@SLUG@@": slug,
        "@@DESCRIPTION@@": description,
        "@@KEYWORDS@@": keywords,
        "@@ASSETS_PATH@@": assets_path,
        "@@HOME_PATH@@": home_path,
        "@@SNIPPETS_URL@@": snippets_url,
        "@@SOURCE_URL@@": source_url,
        "@@LANG_DROPDOWN_HTML@@": lang_dropdown_html,
        "@@BADGE_VERSION@@": badge_version,
        "@@NAV_HOME_TEXT@@": nav_home_text,
        "@@NAV_REPO_TEXT@@": nav_repo_text,
        "@@NAV_UPSTREAM_TEXT@@": nav_upstream_text,
        "@@NAV_SOURCE_TEXT@@": nav_source_text,
        "@@THEME_TIP@@": theme_tip,
        "@@CHAPTER_BADGE@@": chapter_badge,
        "@@TOC_LABEL@@": toc_label,
        "@@TOC_HTML@@": toc_html,
        "@@BREADCRUMB_ROOT@@": breadcrumb_root,
        "@@BREADCRUMB_CH1@@": breadcrumb_ch1,
        "@@ALL_CHAPTERS_HTML@@": all_chapters_html,
        "@@FOOTER_CTA_TITLE@@": footer_cta_title,
        "@@FOOTER_CTA_DESC@@": footer_cta_desc,
        "@@FOOTER_CTA_STAR@@": footer_cta_star,
        "@@FOOTER_CTA_BROWSE@@": footer_cta_browse,
        "@@DEFAULT_FILE@@": default_file,
        "@@CODE_PANE_LINES@@": code_pane_lines,
        "@@COPY_SNIPPET_TITLE@@": copy_snippet_title,
        "@@COPY_SNIPPET_TEXT@@": copy_snippet_text,
        "@@COPY_ALL_TITLE@@": copy_all_title,
        "@@COPY_ALL_TEXT@@": copy_all_text,
        "@@LOCATE_TITLE@@": locate_title,
        "@@LOCATE_TEXT@@": locate_text,
        "@@LOADING_MSG@@": loading_msg,
        "@@COMMIT_ANCHOR_TEXT@@": commit_anchor_text,
        "@@REPO@@": repo,
        "@@BRANCH@@": meta.get("branch", "main"),
        "@@COMMIT@@": commit,
        "@@VERSION@@": meta.get("version", "v1.0.0"),
        "@@STARS@@": stars
    }
    for rep_k, rep_v in replacements.items():
        rendered_page = rendered_page.replace(rep_k, rep_v)

    target_html = os.path.join(book_dir, "index.html")
    with open(target_html, "w", encoding="utf-8") as f:
        f.write(rendered_page)
    print(f"    [+] Wrote ZH {target_html} ({len(rendered_page)} bytes)")

    # Also write books/{slug}.html for backward compatibility
    slug_html = os.path.join(BOOKS_DIR, f"{slug}.html")
    page_for_slug_html = (
        rendered_page
        .replace("../../assets/", "../assets/")
        .replace("../../index.html", "../index.html")
        .replace("./snippets.json", f"./{slug}/snippets.json")
    )
    with open(slug_html, "w", encoding="utf-8") as f:
        f.write(page_for_slug_html)

    return len(ch_files)

def render_english_edition(slug, book_dir, meta):
    """Render the genuine English edition using website/books/en/{slug}.html."""
    src_en_html = os.path.join(WEBSITE_EN_DIR, f"{slug}.html")
    if not os.path.exists(src_en_html):
        print(f"    [-] Warning: Translated English source not found: {src_en_html}")
        return 0

    soup = BeautifulSoup(open(src_en_html, encoding='utf-8'), 'html.parser')
    sections = soup.find_all('section', class_=re.compile(r'reader-chapter-section'))
    if not sections:
        print(f"    [-] Warning: No sections found in {src_en_html}")
        return 0

    repo = meta.get("repo", "")
    commit = meta.get("commit", meta.get("branch", "main"))
    default_file = meta.get("defaultFile", "README.md")
    stars = meta.get("stars", "10k+")
    title = meta.get("title", meta.get("titleEn", slug))
    summary = meta.get("summary", meta.get("summaryEn", ""))

    page_title = f'"{title}" · Architecture Deep Dive | AiReadCodeBooks'
    html_lang = "en"
    assets_path = "../../../assets/"
    home_path = "../../../index.html"
    snippets_url = "../snippets.json"
    source_url = f"https://github.com/maleRjc/AiReadCodeBooks/tree/main/books/{slug}/en"
    badge_version = "Online Book"
    nav_home_text = "Bookstore Lobby"
    nav_repo_text = "GitHub Repo"
    nav_upstream_text = f"Upstream: {repo}"
    nav_source_text = "View Markdown Source"
    theme_tip = "Toggle Dark/Light Mode"
    chapter_badge = f"{len(sections)} Chapters · Complete Deep Dive"
    toc_label = "Table of Contents"
    breadcrumb_root = "Bookstore"
    breadcrumb_ch1 = "Chapter 01"
    footer_cta_title = "To understand any complex project, all you really need is a good book"
    footer_cta_desc = "This book was automatically compiled by AiReadCode by scanning the official repository, with real commit line numbers permanently anchored."
    footer_cta_star = "Star GitHub Repo ★"
    footer_cta_browse = "Browse More Books →"
    code_pane_lines = "Overview"
    copy_snippet_title = "Copy highlighted code slice (or full file if none)"
    copy_snippet_text = "Copy snippet"
    copy_all_title = "Copy full source file"
    copy_all_text = "Copy all"
    locate_title = "Scroll to highlighted lines"
    locate_text = "Locate"
    loading_msg = "⚡ Loading source file..."
    commit_anchor_text = "Commit Permanently Immutable Anchored"
    keywords = f"{repo}, {title}, Source Code Walkthrough, Architecture Analysis, AiReadCode, GitHub Pages"

    lang_dropdown_html = f"""        <div class="lang-dropdown" id="lang-dropdown">
          <button class="lang-dropdown-btn" type="button" aria-haspopup="true" onclick="window.toggleLangMenu(event)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
            <span>🇺🇸 English</span>
            <svg class="chevron-icon" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
          <div class="lang-dropdown-menu" id="lang-menu">
            <a href="../index.html#ch-01" class="lang-dropdown-item" data-lang="zh" onclick="window.switchLanguage('zh', event)">🇨🇳 简体中文</a>
            <a href="#ch-01" class="lang-dropdown-item active" data-lang="en" onclick="window.switchLanguage('en', event)">🇺🇸 English</a>
          </div>
        </div>"""

    toc_items = []
    rendered_chapters = []
    en_dir = os.path.join(book_dir, "en")
    os.makedirs(en_dir, exist_ok=True)

    meta_chapters = meta.get("chapters", [])
    ch_slug_map = {}
    for c in meta_chapters:
        idx = c.get("index", 1)
        ch_slug_map[idx] = c.get("slug", f"chapter-{idx:02d}")

    for idx, sec in enumerate(sections):
        order_str = f"{idx+1:02d}"
        active_cls = " active" if idx == 0 else ""
        active_sec_cls = " active-chapter" if idx == 0 else ""

        # Extract title from h2
        h2_el = sec.find(['h2', 'h1'])
        raw_title = h2_el.get_text() if h2_el else f"Chapter {idx+1}"
        clean_title = clean_repeated_prefix(raw_title, "en")

        toc_items.append(
            f'          <li><a href="#ch-{order_str}" class="toc-link{active_cls}" data-target="ch-{order_str}">'
            f'<span class="toc-num">{order_str}</span><span>{clean_title}</span></a></li>'
        )

        # Extract chapter-body inner HTML
        body_el = sec.find('div', class_='chapter-body')
        body_html = str(body_el) if body_el else ""
        # Clean any inner duplicate h2 in body if present
        body_html = re.sub(r'^\s*<div class="chapter-body">\s*<h2>.*?</h2>', '<div class="chapter-body">', body_html, flags=re.DOTALL)

        prev_btn = f'<a href="#ch-{idx:02d}" class="btn btn-secondary btn-sm btn-ch-nav" data-target="ch-{idx:02d}">← Prev: Chapter {idx}</a>' if idx > 0 else '<span></span>'
        next_btn = f'<a href="#ch-{idx+2:02d}" class="btn btn-primary btn-sm btn-ch-nav" data-target="ch-{idx+2:02d}">Next: Chapter {idx+2} →</a>' if idx < len(sections) - 1 else '<span></span>'

        commit_short = commit[:8] if commit else ""
        sec_html = f'''
      <section class="reader-chapter-section{active_sec_cls}" id="ch-{order_str}" data-title="{clean_title}">
        <div class="chapter-header">
          <span class="chapter-badge">CHAPTER {order_str}</span>
          <h2 class="reader-chapter-title">{clean_title}</h2>
          <div class="reader-chapter-meta">
            <span>Upstream: {repo}</span>
            <span>·</span>
            <span>Commit @{commit_short}</span>
            <span>·</span>
            <span>Progress: Chapter {idx+1} of {len(sections)}</span>
          </div>
        </div>
        {body_html}
        <div class="chapter-nav-bar">
          {prev_btn}
          <a href="#reader-toc" class="btn btn-secondary btn-sm" onclick="window.scrollTo(0, 0); return false;">Back to top ↑</a>
          {next_btn}
        </div>
      </section>
        '''
        rendered_chapters.append(sec_html)

        # Also write clean English markdown to books/{slug}/en/{order_str}-{slug}.md
        ch_slug = ch_slug_map.get(idx+1, f"chapter-{order_str}")
        md_filename = f"{order_str}-{ch_slug}.md"
        md_filepath = os.path.join(en_dir, md_filename)
        try:
            en_md = clean_html_to_markdown(str(sec), clean_title)
            with open(md_filepath, "w", encoding="utf-8") as mf:
                mf.write(en_md)
        except Exception as e:
            pass

    toc_html = "\n".join(toc_items)
    all_chapters_html = "\n".join(rendered_chapters)
    description = summary[:160].replace('"', '&quot;')

    rendered_page = READER_TEMPLATE
    replacements = {
        "@@HTML_LANG@@": html_lang,
        "@@PAGE_TITLE@@": page_title,
        "@@BOOK_TITLE@@": title,
        "@@SLUG@@": slug,
        "@@DESCRIPTION@@": description,
        "@@KEYWORDS@@": keywords,
        "@@ASSETS_PATH@@": assets_path,
        "@@HOME_PATH@@": home_path,
        "@@SNIPPETS_URL@@": snippets_url,
        "@@SOURCE_URL@@": source_url,
        "@@LANG_DROPDOWN_HTML@@": lang_dropdown_html,
        "@@BADGE_VERSION@@": badge_version,
        "@@NAV_HOME_TEXT@@": nav_home_text,
        "@@NAV_REPO_TEXT@@": nav_repo_text,
        "@@NAV_UPSTREAM_TEXT@@": nav_upstream_text,
        "@@NAV_SOURCE_TEXT@@": nav_source_text,
        "@@THEME_TIP@@": theme_tip,
        "@@CHAPTER_BADGE@@": chapter_badge,
        "@@TOC_LABEL@@": toc_label,
        "@@TOC_HTML@@": toc_html,
        "@@BREADCRUMB_ROOT@@": breadcrumb_root,
        "@@BREADCRUMB_CH1@@": breadcrumb_ch1,
        "@@ALL_CHAPTERS_HTML@@": all_chapters_html,
        "@@FOOTER_CTA_TITLE@@": footer_cta_title,
        "@@FOOTER_CTA_DESC@@": footer_cta_desc,
        "@@FOOTER_CTA_STAR@@": footer_cta_star,
        "@@FOOTER_CTA_BROWSE@@": footer_cta_browse,
        "@@DEFAULT_FILE@@": default_file,
        "@@CODE_PANE_LINES@@": code_pane_lines,
        "@@COPY_SNIPPET_TITLE@@": copy_snippet_title,
        "@@COPY_SNIPPET_TEXT@@": copy_snippet_text,
        "@@COPY_ALL_TITLE@@": copy_all_title,
        "@@COPY_ALL_TEXT@@": copy_all_text,
        "@@LOCATE_TITLE@@": locate_title,
        "@@LOCATE_TEXT@@": locate_text,
        "@@LOADING_MSG@@": loading_msg,
        "@@COMMIT_ANCHOR_TEXT@@": commit_anchor_text,
        "@@REPO@@": repo,
        "@@BRANCH@@": meta.get("branch", "main"),
        "@@COMMIT@@": commit,
        "@@VERSION@@": meta.get("version", "v1.0.0"),
        "@@STARS@@": stars
    }
    for rep_k, rep_v in replacements.items():
        rendered_page = rendered_page.replace(rep_k, rep_v)

    target_html = os.path.join(en_dir, "index.html")
    with open(target_html, "w", encoding="utf-8") as f:
        f.write(rendered_page)
    print(f"    [+] Wrote EN {target_html} ({len(rendered_page)} bytes, {len(sections)} chapters)")
    return len(sections)

def compile_all():
    print(f"[*] Starting AiReadCodeBooks compilation in: {BOOKS_DIR}")
    if not os.path.exists(BOOKS_DIR):
        print(f"[-] Error: Books directory not found: {BOOKS_DIR}")
        return

    book_slugs = sorted([d for d in os.listdir(BOOKS_DIR) if os.path.isdir(os.path.join(BOOKS_DIR, d)) and not d.startswith(".")])
    cards_html = []

    for slug in book_slugs:
        book_dir = os.path.join(BOOKS_DIR, slug)
        meta_file = os.path.join(book_dir, "meta.json")
        if not os.path.exists(meta_file):
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"\n=======================================================")
        print(f"[*] Compiling book: {slug} ({meta.get('titleZh', slug)})")
        print(f"=======================================================")

        # 1. Render Chinese Edition (books/{slug}/index.html)
        zh_count = render_chinese_edition(slug, book_dir, meta)

        # 2. Render Genuine English Edition (books/{slug}/en/index.html)
        en_count = render_english_edition(slug, book_dir, meta)

        # 3. Snippets Extraction & Caching
        snippets_json_path = os.path.join(book_dir, "snippets.json")
        snippets = {}
        if os.path.exists(snippets_json_path):
            try:
                with open(snippets_json_path, "r", encoding="utf-8") as sf:
                    snippets = json.load(sf)
            except Exception:
                snippets = {}

        if not snippets:
            local_code_dir = LOCAL_CODE_DIRS.get(slug, "")
            zh_dir = os.path.join(book_dir, "zh")
            snippets = extract_book_snippets(zh_dir, local_code_dir)

            existing_snippet_path = os.path.join(ROOT_DIR, "..", "GitHub", "AiReadCode", "website", "books", f"{slug}-snippets.json")
            if not snippets and os.path.exists(existing_snippet_path):
                with open(existing_snippet_path, "r", encoding="utf-8") as sf:
                    snippets = json.load(sf)

            if snippets:
                with open(snippets_json_path, "w", encoding="utf-8") as f:
                    json.dump(snippets, f, ensure_ascii=False)
                print(f"    [+] Wrote {snippets_json_path} ({len(snippets)} snippets)")

        # 4. Prepare Bookshelf Card
        repo = meta.get("repo", "")
        commit = meta.get("commit", meta.get("branch", "main"))
        title_zh = clean_repeated_prefix(meta.get("titleZh", slug), "zh")
        stars = meta.get("stars", "10k+")
        tags = meta.get("tags", ["源码剖析", "FACT 锚定"])
        summary = meta.get("summaryZh", meta.get("summary", ""))
        tags_html = "".join(f"<span>{t}</span>" for t in tags[:4])

        en_btn = f'<a href="books/{slug}/en/#ch-01" class="btn btn-secondary btn-sm" style="padding: 6px 12px;" title="Read in English"><span>🇺🇸 EN</span></a>' if en_count else ''

        cards_html.append(f'''
      <div class="book-card">
        <span class="book-card-badge">{zh_count or len(meta.get("chapters", []))} 章全集 · 深度专著</span>
        <h2 class="book-card-title">{title_zh}</h2>
        <div class="book-card-repo">{repo} · ★ {stars} · Commit @{commit[:7]}</div>
        <p class="book-card-desc">{summary[:120]}...</p>
        <div class="book-card-tags">
          {tags_html}
        </div>
        <div style="display: flex; gap: 8px; margin-top: auto;">
          <a href="books/{slug}/#ch-01" class="btn btn-primary btn-sm" style="flex: 1; text-align: center;">
            <span>🇨🇳 沉浸式阅读 →</span>
          </a>
          {en_btn}
        </div>
      </div>
        ''')

    # Render root index.html
    root_index_html = BOOKSHELF_TEMPLATE.replace("@@CARDS_HTML@@", "\n".join(cards_html))
    root_index_path = os.path.join(ROOT_DIR, "index.html")
    with open(root_index_path, "w", encoding="utf-8") as f:
        f.write(root_index_html)
    print(f"\n[+] Successfully generated root bookshelf portal: {root_index_path}")
    print("[+] All books and editions compiled successfully!")

if __name__ == "__main__":
    compile_all()

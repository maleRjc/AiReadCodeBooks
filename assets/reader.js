/**
 * AiReadCodeBooks - Standalone Multi-column Interactive Web Reader Engine
 * Zero External Server Dependency · Pinned Commit CDN + Snippets Dual-Engine
 */

(function () {
  'use strict';

  // 1. Theme Switcher (Obsidian Dark / Clean Light)
  const THEME_KEY = 'arc_theme';

  function getSystemTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
  }

  function getSavedTheme() {
    return localStorage.getItem(THEME_KEY) || getSystemTheme();
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btns = document.querySelectorAll('.theme-toggle-btn');
    btns.forEach(btn => {
      const tip = theme === 'light' ? '切换为深色模式' : '切换为浅色模式';
      btn.setAttribute('title', tip);
      btn.setAttribute('aria-label', tip);
    });
    if (typeof reRenderAllMermaidThemes === 'function') {
      reRenderAllMermaidThemes(theme);
    }
  }

  const initialTheme = getSavedTheme();
  document.documentElement.setAttribute('data-theme', initialTheme);

  window.toggleTheme = function () {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'light' ? 'dark' : 'light';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  };

  // 2. Code Pane & FACT Highlighting Engine
  const cachedFiles = {};
  let currentLoadedFile = '';
  let currentTargetLines = '';
  let currentHighlightedCode = '';

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function parseLineRange(linesStr) {
    let start = 0;
    let end = 0;
    if (linesStr) {
      if (linesStr.includes('-')) {
        const parts = linesStr.split('-');
        start = parseInt(parts[0], 10) || 0;
        end = parseInt(parts[1], 10) || start;
      } else {
        start = end = parseInt(linesStr, 10) || 0;
      }
    }
    return { start, end };
  }

  function updateHighlightAndScroll(linesStr) {
    currentTargetLines = linesStr || '';
    const { start, end } = parseLineRange(linesStr);

    const linesBadge = document.getElementById('code-pane-lines');
    if (linesBadge) {
      linesBadge.innerText = linesStr ? ('L' + linesStr) : '全览';
    }

    const body = document.getElementById('code-pane-content');
    if (!body) return;

    const rows = body.querySelectorAll('.code-line');
    let firstTargetRow = null;
    const snippetLines = [];

    rows.forEach(row => {
      const ln = parseInt(row.getAttribute('data-ln'), 10);
      const isH = (start > 0 && end > 0 && ln >= start && ln <= end);
      if (isH) {
        row.classList.add('highlight');
        if (!firstTargetRow) firstTargetRow = row;
        const textSpan = row.querySelector('.code-text');
        if (textSpan) snippetLines.push(textSpan.textContent);
      } else {
        row.classList.remove('highlight');
      }
    });

    currentHighlightedCode = snippetLines.join('\n');

    if (firstTargetRow) {
      firstTargetRow.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }

    body.classList.add('flash-highlight');
    setTimeout(() => { body.classList.remove('flash-highlight'); }, 350);
  }

  function renderFullFile(file, rawText, linesStr) {
    currentLoadedFile = file;
    currentTargetLines = linesStr || '';

    const basename = file.split('/').pop() || file;
    const filenameEl = document.getElementById('code-pane-filename');
    if (filenameEl) {
      filenameEl.innerText = basename;
      filenameEl.title = file;
    }

    const linesEl = document.getElementById('code-pane-lines');
    if (linesEl) {
      linesEl.innerText = linesStr ? ('L' + linesStr) : '全览';
    }

    const filepathEl = document.getElementById('code-pane-filepath');
    if (filepathEl) {
      filepathEl.innerText = file;
      filepathEl.title = file;
    }

    const fileLines = rawText.split('\n').map(l => l.endsWith('\r') ? l.slice(0, -1) : l);
    const statsEl = document.getElementById('code-pane-file-stats');
    if (statsEl) {
      const meta = window.BOOK_META || {};
      const commitShort = meta.commit ? ('@' + meta.commit.slice(0, 7)) : '';
      statsEl.innerText = `全量 ${fileLines.length} 行 ${commitShort}`;
    }

    const { start, end } = parseLineRange(linesStr);
    const body = document.getElementById('code-pane-content');
    if (!body) return;

    const htmlArr = [];
    const snippetLines = [];
    let firstTargetIdx = -1;

    for (let i = 0; i < fileLines.length; i++) {
      const ln = i + 1;
      const isH = (start > 0 && end > 0 && ln >= start && ln <= end);
      if (isH) {
        if (firstTargetIdx === -1) firstTargetIdx = i;
        snippetLines.push(fileLines[i]);
      }
      htmlArr.push(
        `<div class="code-line ${isH ? 'highlight' : ''}" data-ln="${ln}">` +
        `<span class="line-no">${ln}</span>` +
        `<span class="code-text">${escapeHtml(fileLines[i])}</span>` +
        `</div>`
      );
    }

    currentHighlightedCode = snippetLines.join('\n');
    body.innerHTML = htmlArr.join('');

    if (firstTargetIdx >= 0 && body.children[firstTargetIdx]) {
      body.children[firstTargetIdx].scrollIntoView({ block: 'center', behavior: 'smooth' });
    }

    body.classList.add('flash-highlight');
    setTimeout(() => { body.classList.remove('flash-highlight'); }, 350);
  }

  function renderSnippetFallback(file, lines, snippetData) {
    currentLoadedFile = file;
    currentTargetLines = lines || '';

    const basename = file.split('/').pop() || file;
    const filenameEl = document.getElementById('code-pane-filename');
    if (filenameEl) {
      filenameEl.innerText = basename;
      filenameEl.title = file;
    }

    const linesEl = document.getElementById('code-pane-lines');
    if (linesEl) {
      linesEl.innerText = lines ? ('L' + lines) : '切片';
    }

    const filepathEl = document.getElementById('code-pane-filepath');
    if (filepathEl) {
      filepathEl.innerText = file;
      filepathEl.title = file;
    }

    const statsEl = document.getElementById('code-pane-file-stats');
    if (statsEl) {
      statsEl.innerText = '离线切片模式 (已固化)';
    }

    const body = document.getElementById('code-pane-content');
    if (!body) return;

    const items = snippetData.items || [];
    const htmlArr = [];
    const snippetLines = [];
    let firstTargetIdx = -1;

    items.forEach((item, idx) => {
      const ln = item.n;
      const isH = item.h === 1;
      if (isH) {
        if (firstTargetIdx === -1) firstTargetIdx = idx;
        snippetLines.push(item.t);
      }
      htmlArr.push(
        `<div class="code-line ${isH ? 'highlight' : ''}" data-ln="${ln}">` +
        `<span class="line-no">${ln}</span>` +
        `<span class="code-text">${escapeHtml(item.t)}</span>` +
        `</div>`
      );
    });

    currentHighlightedCode = snippetLines.join('\n');
    body.innerHTML = htmlArr.join('');

    if (firstTargetIdx >= 0 && body.children[firstTargetIdx]) {
      body.children[firstTargetIdx].scrollIntoView({ block: 'center', behavior: 'smooth' });
    }

    body.classList.add('flash-highlight');
    setTimeout(() => { body.classList.remove('flash-highlight'); }, 350);
  }

  window.highlightFact = function (file, lines) {
    // 1. Highlight active pills
    document.querySelectorAll('.fact-pill').forEach(el => {
      if (el.getAttribute('data-file') === file && el.getAttribute('data-lines') === lines) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });

    // 2. If same file already loaded, scroll & highlight
    if (currentLoadedFile === file && document.getElementById('code-pane-content').children.length > 0) {
      updateHighlightAndScroll(lines);
      return;
    }

    // 3. Check memory cache
    if (cachedFiles[file]) {
      renderFullFile(file, cachedFiles[file], lines);
      return;
    }

    // 4. Fetch full source with Commit Pinning
    const body = document.getElementById('code-pane-content');
    if (body) {
      body.innerHTML = `<div class="code-loading-msg">⚡ 正在载入全量源文件 (${escapeHtml(file.split('/').pop() || file)})...</div>`;
    }

    const meta = window.BOOK_META || {};
    const repo = meta.repo || '';
    const commit = meta.commit || meta.branch || 'main';

    if (!repo) {
      if (body) {
        body.innerHTML = `<div style="padding: 24px; color: #ef4444; font-size: 12px; font-family: var(--font-mono);">⚠️ 未配置源仓库信息</div>`;
      }
      return;
    }

    // Dual-Engine Fetcher: jsDelivr CDN -> Raw GitHub -> Local Snippets Fallback
    const cdnUrl = `https://cdn.jsdelivr.net/gh/${repo}@${commit}/${file}`;
    const rawGithubUrl = `https://raw.githubusercontent.com/${repo}/${commit}/${file}`;

    function trySnippetsFallback() {
      const key = `${file}:${lines}`;
      if (window.BOOK_SNIPPETS && window.BOOK_SNIPPETS[key]) {
        renderSnippetFallback(file, lines, window.BOOK_SNIPPETS[key]);
        return true;
      }
      return false;
    }

    fetch(cdnUrl)
      .then(res => {
        if (!res.ok) throw new Error(`jsDelivr HTTP ${res.status}`);
        return res.text();
      })
      .then(rawText => {
        cachedFiles[file] = rawText;
        renderFullFile(file, rawText, lines);
      })
      .catch(cdnErr => {
        console.warn('[AiReadCode] CDN fetch failed, trying raw GitHub:', cdnErr);
        fetch(rawGithubUrl)
          .then(res => {
            if (!res.ok) throw new Error(`GitHub HTTP ${res.status}`);
            return res.text();
          })
          .then(rawText => {
            cachedFiles[file] = rawText;
            renderFullFile(file, rawText, lines);
          })
          .catch(rawErr => {
            console.warn('[AiReadCode] Raw GitHub fetch failed, attempting snippet fallback:', rawErr);
            if (trySnippetsFallback()) return;

            if (body) {
              const commitShort = commit.slice(0, 8);
              body.innerHTML = `
                <div style="padding: 24px 18px; color: #f87171; font-size: 12.5px; font-family: var(--font-mono);">
                  <div style="font-weight: 700; margin-bottom: 8px;">⚠️ 源码文件加载失败</div>
                  <div style="color: var(--text-muted); margin-bottom: 4px;">文件: ${escapeHtml(file)}</div>
                  <div style="color: var(--text-muted); margin-bottom: 12px;">版本: Commit @${commitShort}</div>
                  <div style="font-size: 11.5px; color: var(--text-secondary); margin-bottom: 16px;">
                    网络策略阻拦了 CDN 请求。你可以直接在 GitHub 查看本文件：
                  </div>
                  <a href="https://github.com/${repo}/blob/${commit}/${file}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary btn-sm" style="display: inline-flex;">
                    在 GitHub 仓库查看原始文件 ↗
                  </a>
                </div>
              `;
            }
          });
      });
  };

  // 3. Copy Code Handlers
  window.copyCurrentCode = function (btn) {
    const code = currentHighlightedCode || cachedFiles[currentLoadedFile] || '';
    if (!code) return;
    navigator.clipboard.writeText(code).then(() => {
      const span = btn.querySelector('.btn-text') || btn;
      const orig = span.innerText;
      span.innerText = orig.includes('复制') ? '已复制!' : 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        span.innerText = orig;
        btn.classList.remove('copied');
      }, 1800);
    });
  };

  window.copyFullFile = function (btn) {
    const fullText = cachedFiles[currentLoadedFile] || '';
    if (!fullText) return;
    navigator.clipboard.writeText(fullText).then(() => {
      const span = btn.querySelector('.btn-text') || btn;
      const orig = span.innerText;
      span.innerText = orig.includes('复制') ? '已复制全文!' : 'Copied Full!';
      btn.classList.add('copied');
      setTimeout(() => {
        span.innerText = orig;
        btn.classList.remove('copied');
      }, 1800);
    });
  };

  window.locateTargetLines = function () {
    const body = document.getElementById('code-pane-content');
    if (!body) return;
    const highlightedRow = body.querySelector('.code-line.highlight');
    if (highlightedRow) {
      highlightedRow.scrollIntoView({ block: 'center', behavior: 'smooth' });
      body.classList.add('flash-highlight');
      setTimeout(() => { body.classList.remove('flash-highlight'); }, 350);
    }
  };

  // 4. Chapter Switching & TOC Hash Router
  window.switchChapter = function (chId, scroll = true) {
    if (!chId) chId = 'ch-01';
    chId = chId.replace(/^#/, '');

    const allSections = document.querySelectorAll('.reader-chapter-section');
    let targetSection = document.getElementById(chId);
    if (!targetSection) {
      targetSection = allSections[0];
      chId = targetSection ? targetSection.getAttribute('id') : 'ch-01';
    }

    allSections.forEach(sec => {
      sec.classList.remove('active-chapter');
      sec.style.display = 'none';
    });
    if (targetSection) {
      targetSection.classList.add('active-chapter');
      targetSection.style.display = 'block';
    }

    // Update TOC link active state
    document.querySelectorAll('.toc-link').forEach(link => {
      const linkTarget = link.getAttribute('data-target') || link.getAttribute('href').replace(/^#/, '');
      if (linkTarget === chId) {
        link.classList.add('active');
        try { link.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (e) {}
      } else {
        link.classList.remove('active');
      }
    });

    // Update Breadcrumb
    const chTitle = targetSection ? targetSection.getAttribute('data-title') : '';
    const breadcrumb = document.getElementById('breadcrumb-current-chapter');
    if (breadcrumb && chTitle) {
      breadcrumb.innerText = chTitle;
    }

    // Update language switcher links to preserve current chapter anchor
    updateLangLinks(chId);

    // Sync URL hash without jitter
    if (window.location.hash !== '#' + chId) {
      if (history.pushState) {
        history.pushState(null, null, '#' + chId);
      } else {
        window.location.hash = '#' + chId;
      }
    }

    if (scroll) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (typeof renderActiveChapterMermaids === 'function') {
      renderActiveChapterMermaids(chId);
    }
  };

  // 5. Multi-Language Synchronization & Switching Engine
  function updateLangLinks(chId) {
    if (!chId) return;
    const targetHash = '#' + chId.replace(/^#/, '');
    const pathname = window.location.pathname;

    const m = pathname.match(/^(.*?)\/books\/([^/]+)(?:\/([a-zA-Z]{2}(?:-[a-zA-Z]{2})?))?(?:\/index\.html|\/)?$/);
    const curLang = (m && m[3]) ? m[3].toLowerCase() : ((document.documentElement.getAttribute('lang') || 'zh').toLowerCase().split('-')[0]);
    const isRoot = (!m || !m[3]);

    document.querySelectorAll('.lang-dropdown-item').forEach(item => {
      const lang = (item.getAttribute('data-lang') || '').toLowerCase();
      if (!lang) return;

      if (lang === curLang) {
        item.classList.add('active');
        item.setAttribute('href', targetHash);
      } else {
        item.classList.remove('active');
        if (lang === 'zh') {
          item.setAttribute('href', (isRoot ? '' : '../') + 'index.html' + targetHash);
        } else {
          item.setAttribute('href', (isRoot ? (lang + '/') : ('../' + lang + '/')) + 'index.html' + targetHash);
        }
      }
    });
  }

  window.switchLanguage = function (targetLang, e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    targetLang = (targetLang || 'zh').toLowerCase();
    const currentHash = window.location.hash || '#ch-01';
    const chId = currentHash.replace(/^#/, '');
    const pathname = window.location.pathname;

    const m = pathname.match(/^(.*?)\/books\/([^/]+)(?:\/([a-zA-Z]{2}(?:-[a-zA-Z]{2})?))?(?:\/index\.html|\/)?$/);
    if (m) {
      const base = m[1];
      const slug = m[2];
      const curLang = (m[3] || 'zh').toLowerCase();

      if (curLang === targetLang) {
        const dd = document.getElementById('lang-dropdown');
        if (dd) {
          dd.classList.remove('open');
          const btn = dd.querySelector('.lang-dropdown-btn');
          if (btn) btn.setAttribute('aria-expanded', 'false');
        }
        return;
      }

      let targetUrl = '';
      if (targetLang === 'zh') {
        targetUrl = `${base}/books/${slug}/index.html#${chId}`;
      } else {
        targetUrl = `${base}/books/${slug}/${targetLang}/index.html#${chId}`;
      }
      window.location.href = targetUrl;
      return;
    }

    const knownLangs = ['en', 'ja', 'ko', 'zh-tw', 'es', 'fr', 'de', 'ru', 'pt'];
    const parts = pathname.split('/').filter(Boolean);
    const lastPart = (parts[parts.length - 1] || '').toLowerCase();
    const secondLast = (parts[parts.length - 2] || '').toLowerCase();
    let curLang = 'zh';
    if (knownLangs.includes(lastPart)) {
      curLang = lastPart;
    } else if (knownLangs.includes(secondLast)) {
      curLang = secondLast;
    }

    if (curLang === targetLang) {
      const dd = document.getElementById('lang-dropdown');
      if (dd) dd.classList.remove('open');
      return;
    }

    const isRoot = (curLang === 'zh');
    if (targetLang === 'zh') {
      window.location.href = (isRoot ? '' : '../') + 'index.html#' + chId;
    } else {
      window.location.href = (isRoot ? (targetLang + '/') : ('../' + targetLang + '/')) + 'index.html#' + chId;
    }
  };

  // 6. Language Dropdown Toggle
  window.toggleLangMenu = function (e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    const dd = document.getElementById('lang-dropdown');
    if (dd) {
      const isOpen = dd.classList.toggle('open');
      const btn = dd.querySelector('.lang-dropdown-btn');
      if (btn) btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    }
  };

  // 6. Global Delegated Click Handler
  document.addEventListener('click', function (e) {
    const dd = document.getElementById('lang-dropdown');
    if (dd && !dd.contains(e.target)) {
      dd.classList.remove('open');
      const btn = dd.querySelector('.lang-dropdown-btn');
      if (btn) btn.setAttribute('aria-expanded', 'false');
    }

    const tocLink = e.target.closest('.toc-link');
    if (tocLink) {
      e.preventDefault();
      const target = tocLink.getAttribute('data-target') || tocLink.getAttribute('href').replace(/^#/, '');
      if (target) window.switchChapter(target, true);
      return;
    }

    const navBtn = e.target.closest('.btn-ch-nav');
    if (navBtn) {
      e.preventDefault();
      const target = navBtn.getAttribute('data-target') || navBtn.getAttribute('href').replace(/^#/, '');
      if (target) window.switchChapter(target, true);
      return;
    }
  });

  // ==========================================================================
  // 7. Mermaid Architecture & Flowchart Diagrams Engine
  // ==========================================================================
  let mermaidLoaded = false;
  let mermaidLoadingPromise = null;
  const renderedMermaidCharts = new Map(); // widgetEl -> { id, rawCode, chartContainer, sourceContainer, renderedTheme }
  let mermaidLightboxBackdrop = null;

  function loadMermaidLibrary() {
    if (window.mermaid) {
      mermaidLoaded = true;
      return Promise.resolve(window.mermaid);
    }
    if (mermaidLoadingPromise) return mermaidLoadingPromise;

    mermaidLoadingPromise = new Promise((resolve, reject) => {
      let localSrc = 'assets/mermaid.min.js';
      const readerScript = document.querySelector('script[src*="reader.js"]');
      if (readerScript) {
        const src = readerScript.getAttribute('src');
        localSrc = src.replace('reader.js', 'mermaid.min.js');
      }

      function tryCdnFallback() {
        console.warn('[AiReadCode] Local mermaid.min.js missing or failed, using CDN fallback...');
        const cdnScript = document.createElement('script');
        cdnScript.src = 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js';
        cdnScript.onload = () => {
          mermaidLoaded = true;
          resolve(window.mermaid);
        };
        cdnScript.onerror = (err) => {
          console.error('[AiReadCode] Failed to load Mermaid library:', err);
          reject(err);
        };
        document.head.appendChild(cdnScript);
      }

      const script = document.createElement('script');
      script.src = localSrc;
      script.onload = () => {
        if (window.mermaid) {
          mermaidLoaded = true;
          resolve(window.mermaid);
        } else {
          tryCdnFallback();
        }
      };
      script.onerror = tryCdnFallback;
      document.head.appendChild(script);
    });

    return mermaidLoadingPromise;
  }

  function getMermaidConfig(theme) {
    const isDark = theme === 'dark';
    return {
      startOnLoad: false,
      securityLevel: 'loose',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif',
      theme: isDark ? 'dark' : 'default',
      themeVariables: isDark ? {
        darkMode: true,
        background: '#0e121c',
        mainBkg: '#161e2e',
        textColor: '#f1f5f9',
        lineColor: '#38bdf8',
        primaryColor: '#1e293b',
        primaryTextColor: '#f8fafc',
        primaryBorderColor: '#38bdf8',
        secondaryColor: '#334155',
        secondaryTextColor: '#f8fafc',
        secondaryBorderColor: '#64748b',
        tertiaryColor: '#0f172a',
        tertiaryTextColor: '#f8fafc',
        tertiaryBorderColor: '#475569',
        nodeTextColor: '#f8fafc',
        titleColor: '#38bdf8',
        edgeLabelBackground: '#161b22',
        clusterBkg: '#0b0f19',
        clusterBorder: '#0ea5e9',
        defaultLinkColor: '#94a3b8'
      } : {
        darkMode: false,
        background: '#ffffff',
        mainBkg: '#f0f9ff',
        textColor: '#0f172a',
        lineColor: '#0284c7',
        primaryColor: '#e0f2fe',
        primaryTextColor: '#0369a1',
        primaryBorderColor: '#0284c7',
        secondaryColor: '#f1f5f9',
        secondaryTextColor: '#334155',
        secondaryBorderColor: '#94a3b8',
        tertiaryColor: '#ffffff',
        tertiaryTextColor: '#0f172a',
        tertiaryBorderColor: '#cbd5e1',
        nodeTextColor: '#0f172a',
        titleColor: '#0369a1',
        edgeLabelBackground: '#ffffff',
        clusterBkg: '#f8fafc',
        clusterBorder: '#0284c7',
        defaultLinkColor: '#64748b'
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      },
      sequence: {
        useMaxWidth: true,
        showSequenceNumbers: true
      }
    };
  }

  function createMermaidLightboxModal() {
    if (mermaidLightboxBackdrop) return mermaidLightboxBackdrop;
    const backdrop = document.createElement('div');
    backdrop.className = 'mermaid-modal-backdrop';
    backdrop.innerHTML = `
      <div class="mermaid-modal-content" role="dialog" aria-modal="true" aria-label="Mermaid 架构图高清视图">
        <div class="mermaid-modal-header">
          <div class="mermaid-modal-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
            <span>高清架构图检视</span>
          </div>
          <div class="mermaid-modal-actions">
            <button class="mermaid-modal-close-btn" type="button" aria-label="关闭">&times;</button>
          </div>
        </div>
        <div class="mermaid-modal-body"></div>
      </div>
    `;

    backdrop.querySelector('.mermaid-modal-close-btn').addEventListener('click', () => {
      backdrop.classList.remove('open');
    });

    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) backdrop.classList.remove('open');
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && backdrop.classList.contains('open')) {
        backdrop.classList.remove('open');
      }
    });

    document.body.appendChild(backdrop);
    mermaidLightboxBackdrop = backdrop;
    return backdrop;
  }

  function openMermaidLightbox(svgHtml) {
    const backdrop = createMermaidLightboxModal();
    const body = backdrop.querySelector('.mermaid-modal-body');
    if (body) {
      body.innerHTML = svgHtml;
      const svg = body.querySelector('svg');
      if (svg) {
        svg.style.maxWidth = '100%';
        svg.style.maxHeight = '75vh';
        svg.style.height = 'auto';
      }
    }
    backdrop.classList.add('open');
  }

  let mermaidChartCounter = 0;

  function setupMermaidBlock(blockWrapper) {
    if (blockWrapper.getAttribute('data-mermaid-processed')) return;
    blockWrapper.setAttribute('data-mermaid-processed', 'true');

    const codeEl = blockWrapper.querySelector('code');
    if (!codeEl) return;

    let rawCode = (codeEl.textContent || '').trim();
    if (!rawCode) return;

    // Clean any leading/trailing backticks or markdown fences if present
    rawCode = rawCode.replace(/^```(?:mermaid)?\s*\n?/i, '').replace(/\n?```\s*$/i, '').trim();

    mermaidChartCounter++;
    const chartId = 'arc-mermaid-' + mermaidChartCounter;

    const mermaidWidget = document.createElement('div');
    mermaidWidget.className = 'mermaid-block-wrapper';
    mermaidWidget.setAttribute('data-chart-id', chartId);

    mermaidWidget.innerHTML = `
      <div class="mermaid-block-header">
        <div class="mermaid-badge">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
          <span>架构数据流图 (Mermaid)</span>
        </div>
        <div class="mermaid-actions">
          <button class="mermaid-action-btn btn-view-source" type="button" title="查看 Mermaid 源码">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
            <span class="btn-text">源码</span>
          </button>
          <button class="mermaid-action-btn btn-copy-source" type="button" title="复制 Mermaid 代码">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span class="btn-text">复制</span>
          </button>
          <button class="mermaid-action-btn btn-zoom" type="button" title="高清全屏查看">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
            <span class="btn-text">放大</span>
          </button>
        </div>
      </div>
      <div class="mermaid-chart-container" title="双击或点击放大全屏查看">
        <div class="mermaid-loading">
          <span style="display:inline-block;animation:spin 1s linear infinite;">⚡</span>
          <span>正在渲染架构图...</span>
        </div>
      </div>
      <div class="mermaid-source-container">
        <pre class="source-code-block"><code class="language-mermaid">${escapeHtml(rawCode)}</code></pre>
      </div>
    `;

    const chartContainer = mermaidWidget.querySelector('.mermaid-chart-container');
    const sourceContainer = mermaidWidget.querySelector('.mermaid-source-container');
    const viewSrcBtn = mermaidWidget.querySelector('.btn-view-source');
    const copyBtn = mermaidWidget.querySelector('.btn-copy-source');
    const zoomBtn = mermaidWidget.querySelector('.btn-zoom');

    // Toggle source view
    viewSrcBtn.addEventListener('click', () => {
      const isHidden = sourceContainer.style.display === 'none' || !sourceContainer.style.display;
      sourceContainer.style.display = isHidden ? 'block' : 'none';
      viewSrcBtn.classList.toggle('active', isHidden);
    });

    // Copy source
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(rawCode).then(() => {
        const textSpan = copyBtn.querySelector('.btn-text');
        const orig = textSpan.innerText;
        textSpan.innerText = '已复制!';
        copyBtn.classList.add('active');
        setTimeout(() => {
          textSpan.innerText = orig;
          copyBtn.classList.remove('active');
        }, 1800);
      });
    });

    // Zoom diagram
    function handleZoom() {
      const svg = chartContainer.querySelector('svg');
      if (svg) openMermaidLightbox(svg.outerHTML);
    }
    zoomBtn.addEventListener('click', handleZoom);
    chartContainer.addEventListener('dblclick', handleZoom);

    // Replace old code block wrapper with new widget
    blockWrapper.parentNode.replaceChild(mermaidWidget, blockWrapper);

    renderedMermaidCharts.set(mermaidWidget, {
      id: chartId,
      rawCode: rawCode,
      chartContainer: chartContainer,
      renderedTheme: ''
    });
  }

  function renderMermaidWidget(widgetEl, theme) {
    const info = renderedMermaidCharts.get(widgetEl);
    if (!info) return;
    if (info.renderedTheme === theme && info.chartContainer.querySelector('svg')) return;

    const renderId = info.id + '-' + theme + '-' + Math.random().toString(36).slice(2, 6);

    window.mermaid.render(renderId, info.rawCode)
      .then(({ svg }) => {
        info.chartContainer.innerHTML = svg;
        info.renderedTheme = theme;
      })
      .catch((err) => {
        console.warn('[AiReadCode] Mermaid diagram render failed for', info.id, err);
        info.chartContainer.innerHTML = `
          <div class="mermaid-fallback">
            <div style="font-weight: 600; margin-bottom: 4px;">⚠️ 架构图表解析提示</div>
            <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">${escapeHtml(err.message || '语法未能被 Mermaid 成功解析')}</div>
            <button class="btn btn-secondary btn-sm" onclick="this.closest('.mermaid-block-wrapper').querySelector('.btn-view-source').click()">查看源码</button>
          </div>
        `;
        info.renderedTheme = theme;
      });
  }

  function renderActiveChapterMermaids(targetCh) {
    const curTheme = document.documentElement.getAttribute('data-theme') || 'dark';

    // Find all mermaid code blocks across the page
    const allWrappers = document.querySelectorAll('.code-block-wrapper');
    allWrappers.forEach(wrap => {
      const code = wrap.querySelector('code');
      const headerSpan = wrap.querySelector('.code-block-header span');
      const isMermaid = (code && (code.classList.contains('language-mermaid') || code.className.includes('mermaid'))) ||
                        (headerSpan && headerSpan.textContent.trim().toLowerCase() === 'mermaid');
      if (isMermaid) {
        setupMermaidBlock(wrap);
      }
    });

    if (renderedMermaidCharts.size === 0) return;

    loadMermaidLibrary().then(() => {
      window.mermaid.initialize(getMermaidConfig(curTheme));

      const sec = document.getElementById(targetCh);
      renderedMermaidCharts.forEach((info, widgetEl) => {
        // Prioritize widgets in active section or visible elements
        if (!sec || sec.contains(widgetEl) || widgetEl.offsetParent !== null) {
          renderMermaidWidget(widgetEl, curTheme);
        }
      });
    }).catch(err => {
      console.error('[AiReadCode] Mermaid initialization error:', err);
    });
  }

  function reRenderAllMermaidThemes(newTheme) {
    if (!mermaidLoaded || !window.mermaid) return;
    try {
      window.mermaid.initialize(getMermaidConfig(newTheme));
      renderedMermaidCharts.forEach((info, widgetEl) => {
        info.renderedTheme = '';
        if (widgetEl.offsetParent !== null) {
          renderMermaidWidget(widgetEl, newTheme);
        }
      });
    } catch (e) {
      console.warn('[AiReadCode] Error updating Mermaid theme:', e);
    }
  }

  // 8. Global Delegated Click Handler

  // 7. Initialization on DOMContentLoaded
  function initReader() {
    applyTheme(document.documentElement.getAttribute('data-theme') || 'dark');
    document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
      btn.removeEventListener('click', window.toggleTheme);
      btn.addEventListener('click', window.toggleTheme);
    });

    const hash = window.location.hash;
    const targetCh = (hash && hash.length > 1) ? hash.replace(/^#/, '') : 'ch-01';
    window.switchChapter(targetCh, false);
    updateLangLinks(targetCh);

    // Initial Mermaid render
    if (typeof renderActiveChapterMermaids === 'function') {
      renderActiveChapterMermaids(targetCh);
    }

    // Auto load first FACT pill or default file
    setTimeout(() => {
      const sec = document.getElementById(targetCh);
      const firstPill = sec ? sec.querySelector('.fact-pill') : null;
      if (firstPill) {
        const f = firstPill.getAttribute('data-file');
        const l = firstPill.getAttribute('data-lines');
        window.highlightFact(f, l);
      } else {
        const meta = window.BOOK_META || {};
        if (meta.defaultFile) {
          window.highlightFact(meta.defaultFile, '1-30');
        }
      }
    }, 60);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initReader);
  } else {
    initReader();
  }

  window.addEventListener('hashchange', function () {
    const hash = window.location.hash;
    if (hash && hash.length > 1) {
      window.switchChapter(hash, true);
    }
  });
})();

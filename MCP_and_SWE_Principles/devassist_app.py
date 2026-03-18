"""
DevAssist — AI-Powered Code Assistant
A Flask app combining semantic search, refactoring, doc generation, and metrics.
Demonstrates the same patterns behind Claude Code and GitHub Copilot.
"""

from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import hashlib
import time


app = Flask(__name__)


# !pip install openai python-dotenv pandas
import pandas as pd
import os, json, time
from dotenv import load_dotenv
from openai import OpenAI
import textwrap
import requests

import requests, json, textwrap

# Create a session that ignores proxy env vars (bypasses VPN for local Ollama)
SESSION = requests.Session()
SESSION.trust_env = False  # equivalent of ollama.Client(..., trust_env=False)



import truststore
truststore.inject_into_ssl()



def pretty_print(*args):
    text = " ".join(str(arg) for arg in args)
    try:
        print(textwrap.fill(text, width=80))
    except Exception as e:
        print(text)  # fallback to normal print if text is not a string

        

load_dotenv('/Users/shivam13juna/Documents/scaler/iitr_classes/llm_ref/openai_key.env')  # reads .env file in the current directory

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found! "
        "Make sure you have a .env file with: OPENAI_API_KEY=sk-..."
    )

pretty_print("API key loaded successfully.")

MODEL = 'gpt-5-nano'




client = OpenAI(api_key=api_key)
pretty_print("OpenAI client ready.")

print(f"Using model: {MODEL}")

MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# ── In-memory codebase & state ────────────────────────────────

CODEBASE = {
    "math_helpers.py": '''def calc(a,b,op):
    if op=="add": return a+b
    elif op=="sub": return a-b
    elif op=="mul": return a*b
    elif op=="div":
        if b==0: return "err"
        return a/b
    return None

def avg(lst):
    s=0
    for i in lst: s+=i
    return s/len(lst)

def fib(n):
    if n<=1: return n
    return fib(n-1)+fib(n-2)''',

    "string_helpers.py": '''def fmt(s,w):
    r=""
    for i in range(0,len(s),w):
        r+=s[i:i+w]+"\\n"
    return r

def cnt(s,c):
    t=0
    for x in s:
        if x==c: t+=1
    return t

def rev(s): return s[::-1]

def is_pal(s): return s==s[::-1]''',

    "data_helpers.py": '''def flatten(lst):
    r=[]
    for i in lst:
        if type(i)==list: r.extend(flatten(i))
        else: r.append(i)
    return r

def dedup(lst):
    s=set()
    r=[]
    for i in lst:
        if i not in s:
            s.add(i)
            r.append(i)
    return r

def chunk(lst,n):
    return [lst[i:i+n] for i in range(0,len(lst),n)]''',

    "api_helpers.py": '''import json, time

def retry(fn, tries=3):
    for i in range(tries):
        try: return fn()
        except: time.sleep(1)
    return None

def parse_resp(r):
    try: return json.loads(r)
    except: return {"error": "bad json"}

def build_url(base, params):
    p = "&".join(f"{k}={v}" for k,v in params.items())
    return f"{base}?{p}"'''
}

# CLAUDE.md-style rules loaded into every AI call
STYLE_RULES = """
# CLAUDE.md — Team Coding Standards

## Code Style
1. Use Google-style docstrings with Args, Returns, Raises sections
2. Add type hints on parameters and return type
3. Use descriptive variable names — no single letters
4. Raise specific exceptions — never return error strings
5. Add input validation at the start of every function
6. Prefer list comprehensions when clear
7. Use snake_case for functions and variables

## Architecture
- /helpers — pure utility functions (no side effects)
- /api — external API wrappers
"""

# Embedding cache
embedding_cache = {}
metrics_log = []


# ── Helper functions ──────────────────────────────────────────

def get_embedding(text):
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    if key not in embedding_cache:
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        embedding_cache[key] = resp.data[0].embedding
    return embedding_cache[key]


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_codebase(query, top_k=2):
    q_emb = get_embedding(query)
    scores = []
    for fname, code in CODEBASE.items():
        f_emb = get_embedding(f"File: {fname}\n{code}")
        scores.append((fname, cosine_sim(q_emb, f_emb), code))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── API Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/search", methods=["POST"])
def api_search():
    query = request.json.get("query", "")
    start = time.time()
    results = search_codebase(query, top_k=3)
    elapsed = time.time() - start

    metrics_log.append({"action": "search", "latency": elapsed, "query": query})

    return jsonify({
        "results": [{"file": f, "score": round(s, 3), "code": c} for f, s, c in results],
        "latency_ms": round(elapsed * 1000)
    })


@app.route("/api/refactor", methods=["POST"])
def api_refactor():
    data = request.json
    task = data.get("task", "Refactor this code")
    use_style = data.get("use_style", True)
    use_context = data.get("use_context", True)

    start = time.time()

    # Build context
    context_block = ""
    retrieved_files = []
    if use_context:
        results = search_codebase(task, top_k=2)
        for fname, score, code in results:
            context_block += f"\n### {fname}\n```python\n{code}\n```\n"
            retrieved_files.append(fname)

    # Build system prompt (CLAUDE.md pattern)
    sys_prompt = "You are a senior Python engineer. Return ONLY the refactored code."
    if use_style:
        sys_prompt += f"\n\nTeam coding standards (from CLAUDE.md):\n{STYLE_RULES}"

    prompt = f"Task: {task}"
    if context_block:
        prompt += f"\n\nRelevant codebase files:{context_block}"
    prompt += "\n\nRefactor the relevant code."

    response = client.responses.create(
        model=MODEL,
        instructions=sys_prompt,
        input=prompt
    )

    elapsed = time.time() - start

    metrics_log.append({
        "action": "refactor",
        "latency": elapsed,
        "tokens": response.usage.total_tokens,
        "task": task
    })

    return jsonify({
        "refactored_code": response.output_text,
        "context_files": retrieved_files,
        "latency_ms": round(elapsed * 1000),
        "tokens_used": response.usage.total_tokens
    })


@app.route("/api/document", methods=["POST"])
def api_document():
    filename = request.json.get("filename", "")

    if filename not in CODEBASE:
        return jsonify({"error": f"File '{filename}' not found"}), 404

    start = time.time()

    response = client.responses.create(
        model=MODEL,
        instructions="""Generate clean Markdown documentation including:
- Module overview
- Function signatures with parameter descriptions
- Usage examples
Return ONLY the Markdown.""",
        input=f"""Generate documentation for:\n\nFilename: {filename}\n```python\n{CODEBASE[filename]}\n```"""
    )

    elapsed = time.time() - start

    metrics_log.append({
        "action": "document",
        "latency": elapsed,
        "tokens": response.usage.total_tokens,
        "file": filename
    })

    return jsonify({
        "documentation": response.output_text,
        "latency_ms": round(elapsed * 1000),
        "tokens_used": response.usage.total_tokens
    })


@app.route("/api/metrics")
def api_metrics():
    if not metrics_log:
        return jsonify({"message": "No operations recorded yet"})

    total_ops = len(metrics_log)
    avg_latency = sum(m["latency"] for m in metrics_log) / total_ops
    total_tokens = sum(m.get("tokens", 0) for m in metrics_log)
    by_action = {}
    for m in metrics_log:
        action = m["action"]
        by_action.setdefault(action, []).append(m["latency"])

    return jsonify({
        "total_operations": total_ops,
        "average_latency_ms": round(avg_latency * 1000),
        "total_tokens": total_tokens,
        "cache_size": len(embedding_cache),
        "by_action": {
            action: {
                "count": len(latencies),
                "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000)
            }
            for action, latencies in by_action.items()
        }
    })


@app.route("/api/files")
def api_files():
    return jsonify({"files": list(CODEBASE.keys())})


# ── HTML Template ─────────────────────────────────────────────

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DevAssist — AI Code Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: "SF Mono", "Fira Code", "Consolas", monospace;
            background: #0d1117; color: #c9d1d9;
            padding: 20px; max-width: 1200px; margin: 0 auto;
        }
        h1 { color: #58a6ff; margin-bottom: 5px; font-size: 1.6em; }
        .subtitle { color: #8b949e; margin-bottom: 25px; font-size: 0.9em; }
        .tabs {
            display: flex; gap: 0; margin-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }
        .tab {
            padding: 10px 20px; cursor: pointer; color: #8b949e;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab:hover { color: #c9d1d9; }
        .tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
        .panel { display: none; }
        .panel.active { display: block; }
        .input-group { margin-bottom: 15px; }
        label { display: block; color: #8b949e; margin-bottom: 5px; font-size: 0.85em; }
        input[type="text"], textarea, select {
            width: 100%; padding: 10px; background: #161b22;
            border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px;
            font-family: inherit; font-size: 0.9em;
        }
        textarea { min-height: 80px; resize: vertical; }
        button {
            padding: 10px 20px; background: #238636; color: white;
            border: none; border-radius: 6px; cursor: pointer;
            font-family: inherit; font-weight: 600; font-size: 0.9em;
            transition: background 0.2s;
        }
        button:hover { background: #2ea043; }
        button:disabled { background: #21262d; cursor: wait; }
        .toggle-row { display: flex; gap: 20px; margin-bottom: 15px; }
        .toggle-row label { display: flex; align-items: center; gap: 6px; cursor: pointer; color: #c9d1d9; }
        .result-box {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 6px; padding: 15px; margin-top: 15px;
            white-space: pre-wrap; font-size: 0.85em;
            max-height: 500px; overflow-y: auto;
        }
        .meta { color: #8b949e; font-size: 0.8em; margin-top: 8px; }
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-top: 15px;
        }
        .metric-card {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 6px; padding: 15px; text-align: center;
        }
        .metric-value { font-size: 1.8em; color: #58a6ff; font-weight: 700; }
        .metric-label { color: #8b949e; font-size: 0.8em; margin-top: 4px; }
        .search-result {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 6px; padding: 12px; margin-top: 10px;
        }
        .search-result .fname { color: #58a6ff; font-weight: 600; }
        .search-result .score { color: #3fb950; float: right; }
        .search-result pre { margin-top: 8px; font-size: 0.82em; color: #8b949e; }
    </style>
</head>
<body>
    <h1>DevAssist</h1>
    <div class="subtitle">AI-Powered Code Assistant &mdash; Semantic Search &bull; Smart Refactor &bull; Doc Gen &bull; Metrics</div>

    <div class="tabs">
        <div class="tab active" onclick="switchTab('search')">Search</div>
        <div class="tab" onclick="switchTab('refactor')">Refactor</div>
        <div class="tab" onclick="switchTab('docs')">Generate Docs</div>
        <div class="tab" onclick="switchTab('metrics')">Metrics</div>
    </div>

    <!-- SEARCH -->
    <div id="panel-search" class="panel active">
        <div class="input-group">
            <label>Describe what you\'re looking for in plain English</label>
            <input type="text" id="search-query" placeholder="e.g. function to retry API calls">
        </div>
        <button onclick="doSearch()">Search Codebase</button>
        <div id="search-results"></div>
    </div>

    <!-- REFACTOR -->
    <div id="panel-refactor" class="panel">
        <div class="input-group">
            <label>What do you want to refactor?</label>
            <textarea id="refactor-task" placeholder="e.g. Clean up the math helper functions — add type hints and docstrings"></textarea>
        </div>
        <div class="toggle-row">
            <label><input type="checkbox" id="use-context" checked> Use semantic context</label>
            <label><input type="checkbox" id="use-style" checked> Enforce CLAUDE.md style rules</label>
        </div>
        <button onclick="doRefactor()">Refactor</button>
        <div id="refactor-results"></div>
    </div>

    <!-- DOCS -->
    <div id="panel-docs" class="panel">
        <div class="input-group">
            <label>Select a file to document</label>
            <select id="doc-file"></select>
        </div>
        <button onclick="doDocs()">Generate Docs</button>
        <div id="doc-results"></div>
    </div>

    <!-- METRICS -->
    <div id="panel-metrics" class="panel">
        <button onclick="doMetrics()">Refresh Metrics</button>
        <div id="metrics-results"></div>
    </div>

    <script>
        // Tab switching
        function switchTab(name) {
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
            event.target.classList.add("active");
            document.getElementById("panel-" + name).classList.add("active");
        }

        // Load file list
        fetch("/api/files").then(r => r.json()).then(data => {
            const sel = document.getElementById("doc-file");
            data.files.forEach(f => {
                const opt = document.createElement("option");
                opt.value = f; opt.textContent = f;
                sel.appendChild(opt);
            });
        });

        async function doSearch() {
            const query = document.getElementById("search-query").value;
            const btn = event.target; btn.disabled = true; btn.textContent = "Searching...";
            const res = await fetch("/api/search", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({query})
            });
            const data = await res.json();
            btn.disabled = false; btn.textContent = "Search Codebase";
            let html = `<div class="meta">Completed in ${data.latency_ms}ms</div>`;
            data.results.forEach(r => {
                html += `<div class="search-result">
                    <span class="fname">${r.file}</span>
                    <span class="score">Score: ${r.score}</span>
                    <pre>${r.code}</pre>
                </div>`;
            });
            document.getElementById("search-results").innerHTML = html;
        }

        async function doRefactor() {
            const task = document.getElementById("refactor-task").value;
            const use_context = document.getElementById("use-context").checked;
            const use_style = document.getElementById("use-style").checked;
            const btn = event.target; btn.disabled = true; btn.textContent = "Refactoring...";
            const res = await fetch("/api/refactor", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({task, use_context, use_style})
            });
            const data = await res.json();
            btn.disabled = false; btn.textContent = "Refactor";
            let html = `<div class="meta">
                Context from: ${data.context_files.join(", ") || "none"} &bull;
                ${data.latency_ms}ms &bull; ${data.tokens_used} tokens
            </div>`;
            html += `<div class="result-box">${data.refactored_code}</div>`;
            document.getElementById("refactor-results").innerHTML = html;
        }

        async function doDocs() {
            const filename = document.getElementById("doc-file").value;
            const btn = event.target; btn.disabled = true; btn.textContent = "Generating...";
            const res = await fetch("/api/document", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({filename})
            });
            const data = await res.json();
            btn.disabled = false; btn.textContent = "Generate Docs";
            let html = `<div class="meta">${data.latency_ms}ms &bull; ${data.tokens_used} tokens</div>`;
            html += `<div class="result-box">${data.documentation}</div>`;
            document.getElementById("doc-results").innerHTML = html;
        }

        async function doMetrics() {
            const res = await fetch("/api/metrics");
            const data = await res.json();
            if (data.message) {
                document.getElementById("metrics-results").innerHTML =
                    `<div class="result-box">${data.message}</div>`;
                return;
            }
            let html = `<div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${data.total_operations}</div>
                    <div class="metric-label">Total Operations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.average_latency_ms}ms</div>
                    <div class="metric-label">Avg Latency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.total_tokens}</div>
                    <div class="metric-label">Total Tokens</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.cache_size}</div>
                    <div class="metric-label">Cached Embeddings</div>
                </div>
            </div>`;
            if (data.by_action) {
                html += `<div class="result-box">Breakdown by action:\\n`;
                for (const [action, stats] of Object.entries(data.by_action)) {
                    html += `  ${action}: ${stats.count} ops, avg ${stats.avg_latency_ms}ms\\n`;
                }
                html += `</div>`;
            }
            document.getElementById("metrics-results").innerHTML = html;
        }
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    print("Starting DevAssist on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# COLORS: Moss=#2563eb, FAISS=#16a34a, Pinecone=#7c3aed
colors = {
    "Moss": "#2563eb",
    "FAISS": "#16a34a",
    "Pinecone": "#7c3aed"
}

# this took forever to get the colors right


def generate_graphs():
    with open("results/results.json", "r") as f:
        data = json.load(f)

    b_data = data["benchmarks"]
    sys_names = [b["system"] for b in b_data]
    p50s = [b["p50"] for b in b_data]
    p99s = [b["p99"] for b in b_data]
    recalls = [b["recall_at_5"] for b in b_data]
    idx_times = [b["index_time_s"] for b in b_data]
    bar_colors = [colors[s] for s in sys_names]

    # Graph 1: Latency Bar (log scale so faiss/moss aren't squashed to nothing)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(sys_names))
    width = 0.35
    ax.bar(x - width/2, p50s, width, label='P50', color=bar_colors, alpha=0.85)
    ax.bar(x + width/2, p99s, width, label='P99', color=bar_colors, alpha=0.55, hatch='//')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(sys_names)
    ax.set_ylabel('Latency (ms) — log scale')
    ax.set_title('Query Latency Comparison (P50 vs P99)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.2, axis='y')
    for rect, val in zip(ax.patches, p50s + p99s):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() * 1.1,
                f"{val:.2f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig('results/graph_latency_bar.png')
    plt.close(fig)

    # Graph 2: Latency CDF
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for b in b_data:
        sys = b["system"]
        lats = sorted(b["latencies_ms"])
        p = np.linspace(0, 1, len(lats))
        ax.plot(lats, p, label=sys, color=colors[sys], linewidth=2)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Latency CDF — 100 runs × 20 queries')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    fig.savefig('results/graph_latency_cdf.png')
    plt.close(fig)

    # Graph 3: Recall
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y_pos = np.arange(len(sys_names))
    ax.barh(y_pos, recalls, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sys_names)
    ax.set_xlabel('Recall@5 (%)')
    ax.set_title('Retrieval Quality: Recall@5')
    ax.set_xlim(0, 105)
    ax.axvline(x=80, color='r', linestyle='--', alpha=0.4)
    for i, v in enumerate(recalls):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig('results/graph_recall.png')
    plt.close(fig)

    # Graph 4: Scale Curve
    sc = data.get("scale_curve", {})
    if sc and "sizes" in sc:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for sys in ["FAISS", "Pinecone", "Moss"]:
            if sys in sc:
                ax.plot(sc["sizes"], sc[sys], marker='o', label=sys, color=colors[sys], linewidth=2)
        ax.set_xlabel('Corpus Size (Chunks)')
        ax.set_ylabel('P50 Latency (ms) — log scale')
        ax.set_yscale('log')
        ax.set_title('Latency vs Corpus Size')
        ax.legend()
        ax.grid(True, which='both', alpha=0.2)
        fig.tight_layout()
        fig.savefig('results/graph_scale_curve.png')
        plt.close(fig)

    # Graph 5: Indexing Time
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(sys_names, idx_times, color=bar_colors)
    for i, v in enumerate(idx_times):
        ax.text(i, v + 0.5, f"{v:.1f}s", ha='center', fontsize=9)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Indexing Time Comparison')
    fig.tight_layout()
    fig.savefig('results/graph_indexing_time.png')
    plt.close(fig)


def generate_dashboard():
    with open("results/results.json", "r") as f:
        json_str = f.read()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector DB Benchmark: Moss vs Pinecone vs FAISS</title>
    <meta name="description" content="Head-to-head latency and recall benchmark: Moss vs Pinecone vs FAISS">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #0a0f1e;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 2rem 1rem;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 2.5rem; }}
        header h1 {{
            font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #818cf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        header p {{ color: #64748b; font-size: 0.9rem; }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem; margin-bottom: 2rem;
        }}
        .card {{
            background: #1e293b; border-radius: 10px;
            padding: 1.25rem 1.5rem;
            border-top: 3px solid var(--accent);
            transition: transform 0.15s;
        }}
        .card:hover {{ transform: translateY(-2px); }}
        .card h3 {{ font-size: 1rem; color: #94a3b8; margin-bottom: 0.75rem; }}
        .card .big {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
        .card .metric {{ font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem; }}
        .controls {{
            display: flex; gap: 0.5rem; justify-content: center;
            align-items: center; flex-wrap: wrap; margin-bottom: 1rem;
        }}
        button {{
            background: #1e293b; color: #94a3b8;
            border: 1px solid #334155; padding: 0.45rem 1rem;
            border-radius: 6px; cursor: pointer; font-size: 0.85rem;
            transition: all 0.15s;
        }}
        button:hover {{ background: #2d3f55; color: #e2e8f0; }}
        button.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
        .scale-toggle {{
            display: flex; align-items: center; gap: 0.4rem;
            font-size: 0.82rem; color: #64748b; margin-left: 0.5rem;
        }}
        .scale-toggle input {{ cursor: pointer; }}
        .chart-wrap {{
            background: #1e293b; border-radius: 10px; padding: 1.5rem;
        }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; font-size: 0.88rem; }}
        th, td {{ padding: 0.65rem 1rem; text-align: left; border-bottom: 1px solid #0f172a; }}
        th {{ color: #64748b; font-weight: 500; background: #0f172a; }}
        tr:hover td {{ background: #1e293b; }}
        .tag {{
            display: inline-block; padding: 0.1rem 0.45rem;
            border-radius: 4px; font-size: 0.75rem; font-weight: 600;
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Moss vs Pinecone vs FAISS</h1>
        <p>Semantic search benchmark &mdash; Pride and Prejudice &middot; all-MiniLM-L6-v2 &middot; 100 runs &times; 20 queries</p>
    </header>

    <div class="cards" id="cards"></div>

    <div class="controls">
        <button class="active" id="btn-latency"  onclick="showTab('latency')">Latency</button>
        <button id="btn-quality"  onclick="showTab('quality')">Recall@5</button>
        <button id="btn-scale"    onclick="showTab('scale')">Scale Curve</button>
        <button id="btn-indexing" onclick="showTab('indexing')">Index Time</button>
        <label class="scale-toggle">
            <input type="checkbox" id="logToggle" onchange="renderChart()"> Log scale
        </label>
    </div>

    <div class="chart-wrap">
        <canvas id="mainChart"></canvas>
    </div>

    <table id="tbl"></table>
</div>

<script>
const DATA = {json_str};
const COLORS = {{ Moss: "#2563eb", FAISS: "#16a34a", Pinecone: "#7c3aed" }};
let chart = null;
let tab = "latency";

// cards
document.getElementById("cards").innerHTML = DATA.benchmarks.map(b => `
    <div class="card" style="--accent:${{COLORS[b.system]}}">
        <h3>${{b.system}}</h3>
        <div class="big">${{b.p50.toFixed(2)}} ms</div>
        <div class="metric">P99: ${{b.p99.toFixed(2)}} ms</div>
        <div class="metric">Recall@5: ${{b.recall_at_5.toFixed(1)}}%</div>
        <div class="metric">Index: ${{b.index_time_s.toFixed(1)}}s</div>
    </div>`).join("");

// summary table
document.getElementById("tbl").innerHTML =
    "<tr><th>System</th><th>P50 (ms)</th><th>P99 (ms)</th><th>Recall@5</th><th>Index Time</th></tr>" +
    DATA.benchmarks.map(b => `
        <tr>
            <td><span class="tag" style="background:${{COLORS[b.system]}}22;color:${{COLORS[b.system]}}">${{b.system}}</span></td>
            <td>${{b.p50.toFixed(2)}}</td><td>${{b.p99.toFixed(2)}}</td>
            <td>${{b.recall_at_5.toFixed(1)}}%</td><td>${{b.index_time_s.toFixed(1)}}s</td>
        </tr>`).join("");

function showTab(t) {{
    tab = t;
    ["latency","quality","scale","indexing"].forEach(id =>
        document.getElementById("btn-" + id).classList.toggle("active", id === t));
    renderChart();
}}

function renderChart() {{
    const log = document.getElementById("logToggle").checked;
    const yAxis = {{
        type: log ? "logarithmic" : "linear",
        beginAtZero: !log,
        grid: {{ color: "rgba(255,255,255,0.06)" }},
        ticks: {{ color: "#64748b" }}
    }};
    const xAxis = {{ grid: {{ color: "rgba(255,255,255,0.06)" }}, ticks: {{ color: "#64748b" }} }};

    if (chart) chart.destroy();
    const ctx = document.getElementById("mainChart").getContext("2d");
    const names = DATA.benchmarks.map(b => b.system);
    const bcolors = names.map(s => COLORS[s]);

    if (tab === "latency") {{
        chart = new Chart(ctx, {{
            type: "bar",
            data: {{
                labels: names,
                datasets: [
                    {{ label: "P50 (ms)", data: DATA.benchmarks.map(b => b.p50), backgroundColor: bcolors.map(c => c + "cc") }},
                    {{ label: "P99 (ms)", data: DATA.benchmarks.map(b => b.p99), backgroundColor: bcolors.map(c => c + "55"), borderColor: bcolors, borderWidth: 1 }}
                ]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: "#94a3b8" }} }} }}, scales: {{ x: xAxis, y: yAxis }} }}
        }});
    }} else if (tab === "quality") {{
        chart = new Chart(ctx, {{
            type: "bar",
            data: {{ labels: names, datasets: [{{ label: "Recall@5 (%)", data: DATA.benchmarks.map(b => b.recall_at_5), backgroundColor: bcolors }}] }},
            options: {{ indexAxis: "y", responsive: true, plugins: {{ legend: {{ labels: {{ color: "#94a3b8" }} }} }}, scales: {{ x: {{ ...xAxis, max: 100, beginAtZero: true }}, y: {{ ...yAxis, type: "category" }} }} }}
        }});
    }} else if (tab === "scale") {{
        const sc = DATA.scale_curve || {{}};
        chart = new Chart(ctx, {{
            type: "line",
            data: {{
                labels: sc.sizes || [],
                datasets: ["FAISS","Pinecone","Moss"].filter(s => sc[s]).map(s => ({{
                    label: s, data: sc[s], borderColor: COLORS[s],
                    backgroundColor: COLORS[s] + "22", fill: false, tension: 0.1, pointRadius: 4
                }}))
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: "#94a3b8" }} }} }}, scales: {{ x: xAxis, y: yAxis }} }}
        }});
    }} else if (tab === "indexing") {{
        chart = new Chart(ctx, {{
            type: "bar",
            data: {{ labels: names, datasets: [{{ label: "Index Time (s)", data: DATA.benchmarks.map(b => b.index_time_s), backgroundColor: bcolors }}] }},
            options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: "#94a3b8" }} }} }}, scales: {{ x: xAxis, y: yAxis }} }}
        }});
    }}
}}

// start with log on so latency chart is actually readable
document.getElementById("logToggle").checked = true;
renderChart();
</script>
</body>
</html>"""

    with open("results/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    if not os.path.exists("results/results.json"):
        print("error: results/results.json not found. run benchmark first!")
    else:
        generate_graphs()
        generate_dashboard()
        print("Generated 5 graphs + interactive dashboard")
        print("Open results/dashboard.html in your browser")

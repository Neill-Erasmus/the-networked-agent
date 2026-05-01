from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class VisualizationResult:
    output_dir: str
    html_path: str
    got_json_path: str
    rag_json_path: str
    got_dot_path: str
    rag_dot_path: str


class QueryVisualizer:
    """Saves per-query GoT and GraphRAG artifacts and a browser-friendly HTML view."""

    def __init__(self, base_dir: str = "data/visualizations") -> None:
        self.base_dir = base_dir

    @staticmethod
    def _slugify(text: str, max_len: int = 60) -> str:
        value = (text or "").strip().lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        if not value:
            value = "query"
        return value[:max_len]

    @staticmethod
    def _clip(text: str, max_len: int = 120) -> str:
        cleaned = " ".join((text or "").strip().split())
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3] + "..."

    @staticmethod
    def _escape_dot(text: str) -> str:
        return (text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    @staticmethod
    def _script_json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    def _build_got_payload(self, got_snapshot: dict[str, Any], reasoning_path: list[str]) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for node in got_snapshot.get("nodes", []):
            content = str(node.get("content", ""))
            nodes.append(
                {
                    "id": node.get("node_id", ""),
                    "label": self._clip(content, 80),
                    "content": content,
                    "depth": int(node.get("depth", 0)),
                    "score": float(node.get("score", 0.0)),
                    "rationale": str(node.get("rationale", "")),
                }
            )

        edges: list[dict[str, str]] = []
        for edge in got_snapshot.get("edges", []):
            edges.append(
                {
                    "source": str(edge.get("source_id", "")),
                    "target": str(edge.get("target_id", "")),
                    "relation": str(edge.get("relation", "expands")),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "reasoning_path": reasoning_path,
        }

    def _build_rag_payload(self, query: str, retrieval: Any, rag_store: Any) -> dict[str, Any]:
        hits = getattr(retrieval, "hits", [])
        selected_chunk_ids = [hit.chunk_id for hit in hits]
        selected_set = set(selected_chunk_ids)

        nodes: list[dict[str, Any]] = [
            {
                "id": "query",
                "label": self._clip(query, 100),
                "text": query,
                "depth": -1,
                "score": 1.0,
                "source": "user",
            }
        ]

        for hit in hits:
            chunk = rag_store.chunks.get(hit.chunk_id)
            chunk_text = chunk.text if chunk else hit.text
            nodes.append(
                {
                    "id": hit.chunk_id,
                    "label": self._clip(chunk_text, 80),
                    "text": chunk_text,
                    "depth": int(hit.depth),
                    "score": float(hit.score),
                    "source": hit.source,
                }
            )

        edges: list[dict[str, Any]] = []
        seed_ids = [hit.chunk_id for hit in hits if int(hit.depth) == 0]
        if not seed_ids:
            seed_ids = selected_chunk_ids[: min(3, len(selected_chunk_ids))]
        for chunk_id in seed_ids:
            edges.append({"source": "query", "target": chunk_id, "relation": "retrieved"})

        seen_pairs: set[tuple[str, str]] = set()
        for left in selected_chunk_ids:
            neighbors = rag_store.chunk_neighbors.get(left, set())
            for right in neighbors:
                if right not in selected_set:
                    continue
                pair = tuple(sorted((left, right)))
                if left == right or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edges.append({"source": left, "target": right, "relation": "neighbor"})

        hit_table = [
            {
                "chunk_id": hit.chunk_id,
                "source": hit.source,
                "score": float(hit.score),
                "depth": int(hit.depth),
                "text": hit.text,
            }
            for hit in hits
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "hits": hit_table,
            "context_text": getattr(retrieval, "context_text", ""),
        }

    def _to_got_dot(self, got_payload: dict[str, Any]) -> str:
        lines = ["digraph GoT {", "  rankdir=TB;", "  node [shape=box, style=rounded];"]
        for node in got_payload.get("nodes", []):
            label = f"{node['id']} | d={node['depth']} s={node['score']:.2f}\\n{self._clip(node['content'], 90)}"
            lines.append(f'  "{self._escape_dot(node["id"])}" [label="{self._escape_dot(label)}"];')
        for edge in got_payload.get("edges", []):
            lines.append(
                f'  "{self._escape_dot(edge["source"])}" -> "{self._escape_dot(edge["target"])}" '
                f'[label="{self._escape_dot(edge["relation"])}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def _to_rag_dot(self, rag_payload: dict[str, Any]) -> str:
        lines = ["digraph GraphRAG {", "  rankdir=LR;", "  node [shape=ellipse, style=filled, fillcolor=lightgray];"]
        for node in rag_payload.get("nodes", []):
            if node["id"] == "query":
                lines.append('  "query" [shape=box, fillcolor=lightblue, label="query"];')
                continue
            label = f"{node['id']} | depth={node['depth']} score={node['score']:.2f}\\n{self._clip(node['text'], 90)}"
            lines.append(f'  "{self._escape_dot(node["id"])}" [label="{self._escape_dot(label)}"];')
        for edge in rag_payload.get("edges", []):
            lines.append(
                f'  "{self._escape_dot(edge["source"])}" -> "{self._escape_dot(edge["target"])}" '
                f'[label="{self._escape_dot(edge["relation"])}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def _build_html(self, meta: dict[str, Any], got_payload: dict[str, Any], rag_payload: dict[str, Any]) -> str:
        meta_json = self._script_json(meta)
        got_json = self._script_json(got_payload)
        rag_json = self._script_json(rag_payload)

        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Networked Agent Query Visualization</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; background: #f4f6fb; color: #141a26; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    .panel {{ background: #ffffff; border-radius: 14px; padding: 16px; margin-bottom: 16px; box-shadow: 0 6px 22px rgba(20, 26, 38, 0.08); }}
    h1 {{ margin: 0 0 8px 0; }}
    h2 {{ margin: 4px 0 12px 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; font-size: 14px; }}
    .k {{ color: #556070; font-weight: 600; }}
    .v {{ color: #141a26; }}
    .query {{ font-size: 16px; margin-top: 8px; }}
    svg {{ width: 100%; height: 520px; border: 1px solid #d5deec; border-radius: 10px; background: #fbfcff; }}
    .path, .hits {{ margin: 0; padding-left: 20px; }}
    .small {{ color: #5f6e83; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <h1>Networked Agent Query Visualization</h1>
      <div id=\"meta\" class=\"meta\"></div>
      <div id=\"query\" class=\"query\"></div>
    </div>

    <div class=\"grid\">
      <div class=\"panel\">
        <h2>Graph of Thought</h2>
        <svg id=\"gotGraph\"></svg>
        <p class=\"small\">Reasoning Path</p>
        <ol id=\"reasoningPath\" class=\"path\"></ol>
      </div>

      <div class=\"panel\">
        <h2>GraphRAG Retrieval</h2>
        <svg id=\"ragGraph\"></svg>
        <p class=\"small\">Retrieved Chunks</p>
        <ol id=\"hitList\" class=\"hits\"></ol>
      </div>
    </div>
  </div>

  <script>
    const META = {meta_json};
    const GOT = {got_json};
    const RAG = {rag_json};

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function createEl(tag, text) {{
      const el = document.createElement(tag);
      if (text !== undefined) el.textContent = text;
      return el;
    }}

    function renderMeta() {{
      const meta = document.getElementById('meta');
      const query = document.getElementById('query');
      const fields = [
        ['timestamp', META.timestamp],
        ['chat_model', META.chat_model],
        ['embedding_model', META.embedding_model],
        ['artifacts', META.output_dir],
      ];
      for (const [k, v] of fields) {{
        const row = createEl('div');
        row.innerHTML = `<span class=\"k\">${{k}}:</span> <span class=\"v\">${{v}}</span>`;
        meta.appendChild(row);
      }}
      query.textContent = `Query: ${{META.query}}`;
    }}

    function layoutByDepth(nodes) {{
      const byDepth = new Map();
      for (const node of nodes) {{
        const depth = Number(node.depth || 0);
        if (!byDepth.has(depth)) byDepth.set(depth, []);
        byDepth.get(depth).push(node);
      }}
      const depths = [...byDepth.keys()].sort((a, b) => a - b);
      const maxPerDepth = Math.max(1, ...depths.map(d => byDepth.get(d).length));
      const width = Math.max(900, maxPerDepth * 260);
      const height = Math.max(420, depths.length * 170 + 100);
      const coords = new Map();

      depths.forEach((depth, depthIndex) => {{
        const row = byDepth.get(depth);
        const stepX = width / (row.length + 1);
        row.forEach((node, idx) => {{
          coords.set(node.id, {{
            x: (idx + 1) * stepX,
            y: 70 + depthIndex * 160,
          }});
        }});
      }});

      return {{ coords, width, height }};
    }}

    function drawGraph(svgId, data, colorFn) {{
      const svg = document.getElementById(svgId);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (!data.nodes.length) return;

      const layout = layoutByDepth(data.nodes);
      svg.setAttribute('viewBox', `0 0 ${{layout.width}} ${{layout.height}}`);

      for (const edge of data.edges) {{
        const a = layout.coords.get(edge.source);
        const b = layout.coords.get(edge.target);
        if (!a || !b) continue;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', a.x);
        line.setAttribute('y1', a.y);
        line.setAttribute('x2', b.x);
        line.setAttribute('y2', b.y);
        line.setAttribute('stroke', '#8da0be');
        line.setAttribute('stroke-width', '2');
        svg.appendChild(line);
      }}

      for (const node of data.nodes) {{
        const p = layout.coords.get(node.id);
        if (!p) continue;
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', p.x);
        circle.setAttribute('cy', p.y);
        circle.setAttribute('r', '32');
        circle.setAttribute('fill', colorFn(node));
        circle.setAttribute('stroke', '#2f3d57');
        circle.setAttribute('stroke-width', '1.5');
        group.appendChild(circle);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', p.x);
        text.setAttribute('y', p.y - 2);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '12');
        text.setAttribute('font-weight', '700');
        text.textContent = node.id;
        group.appendChild(text);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', p.x);
        label.setAttribute('y', p.y + 52);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '11');
        label.setAttribute('fill', '#2f3d57');
        label.textContent = (node.label || '').slice(0, 36);
        group.appendChild(label);

        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = node.text || node.content || node.label || node.id;
        group.appendChild(title);

        svg.appendChild(group);
      }}
    }}

    function renderLists() {{
      const pathList = document.getElementById('reasoningPath');
      for (const step of GOT.reasoning_path || []) {{
        pathList.appendChild(createEl('li', step));
      }}

      const hitList = document.getElementById('hitList');
      for (const hit of RAG.hits || []) {{
        hitList.appendChild(
          createEl('li', `${{hit.chunk_id}} | score=${{Number(hit.score).toFixed(3)}} | depth=${{hit.depth}} | source=${{hit.source}}`)
        );
      }}
    }}

    renderMeta();
    drawGraph('gotGraph', GOT, (node) => {{
      const s = clamp(Number(node.score || 0), 0, 1);
      const g = Math.round(120 + s * 120);
      return `rgb(88, ${{g}}, 140)`;
    }});
    drawGraph('ragGraph', RAG, (node) => {{
      if (node.id === 'query') return '#7bc6ff';
      const d = Number(node.depth || 0);
      if (d <= 0) return '#8fe8a6';
      if (d === 1) return '#ffd987';
      return '#ffc2aa';
    }});
    renderLists();
  </script>
</body>
</html>
"""

    def save_turn(
        self,
        *,
        query: str,
        answer: str,
        got_snapshot: dict[str, Any],
        reasoning_path: list[str],
        retrieval: Any,
        rag_store: Any,
        chat_model: str,
        embedding_model: str,
    ) -> VisualizationResult:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self._slugify(query)
        output_dir = os.path.join(self.base_dir, f"{timestamp}_{slug}")
        os.makedirs(output_dir, exist_ok=True)

        got_payload = self._build_got_payload(got_snapshot=got_snapshot, reasoning_path=reasoning_path)
        rag_payload = self._build_rag_payload(query=query, retrieval=retrieval, rag_store=rag_store)

        meta_payload = {
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "output_dir": output_dir,
            "retrieved_chunks": [hit.get("chunk_id") for hit in rag_payload.get("hits", [])],
        }

        got_json_path = os.path.join(output_dir, "got_graph.json")
        rag_json_path = os.path.join(output_dir, "graphrag_graph.json")
        got_dot_path = os.path.join(output_dir, "got_graph.dot")
        rag_dot_path = os.path.join(output_dir, "graphrag_graph.dot")
        html_path = os.path.join(output_dir, "visualization.html")
        meta_path = os.path.join(output_dir, "meta.json")

        with open(got_json_path, "w", encoding="utf-8") as outfile:
            json.dump(got_payload, outfile, indent=2, ensure_ascii=False)
        with open(rag_json_path, "w", encoding="utf-8") as outfile:
            json.dump(rag_payload, outfile, indent=2, ensure_ascii=False)
        with open(meta_path, "w", encoding="utf-8") as outfile:
            json.dump(meta_payload, outfile, indent=2, ensure_ascii=False)
        with open(got_dot_path, "w", encoding="utf-8") as outfile:
            outfile.write(self._to_got_dot(got_payload))
        with open(rag_dot_path, "w", encoding="utf-8") as outfile:
            outfile.write(self._to_rag_dot(rag_payload))
        with open(html_path, "w", encoding="utf-8") as outfile:
            outfile.write(self._build_html(meta=meta_payload, got_payload=got_payload, rag_payload=rag_payload))

        return VisualizationResult(
            output_dir=output_dir,
            html_path=html_path,
            got_json_path=got_json_path,
            rag_json_path=rag_json_path,
            got_dot_path=got_dot_path,
            rag_dot_path=rag_dot_path,
        )
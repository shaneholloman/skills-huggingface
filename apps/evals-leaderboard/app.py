from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import requests
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

API_BASE = "https://huggingface.co/api"
PIPELINE_FILTER = "text-generation"
TRENDING_LIMIT = 10
TRENDING_FETCH_LIMIT = 50
PR_SCAN_LIMIT = 40
USER_AGENT = "skills-evals-leaderboard/0.2"
TABLE_HEADERS = [
    "Model",
    "Benchmark",
    "Score",
    "Source",
]

TABLE_DATATYPES = [
    "text",
    "text",
    "number",
    "markdown",
]


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("%"):
            candidate = candidate[:-1]
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    aliases: tuple[str, ...]

    def matches(self, fields: List[str]) -> bool:
        for alias in self.aliases:
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue
            for field in fields:
                if alias_norm in field:
                    return True
        return False


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "mmlu": BenchmarkSpec(
        key="mmlu",
        label="MMLU",
        aliases=("mmlu", "massive multitask language understanding"),
    ),
    "bigcodebench": BenchmarkSpec(
        key="bigcodebench",
        label="BigCodeBench",
        aliases=("bigcodebench", "big code bench"),
    ),
    "arc_mc": BenchmarkSpec(
        key="arc_mc",
        label="ARC MC",
        aliases=(
            "arc mc",
            "arc-challenge",
            "arc challenge",
            "arc multiple choice",
            "arc c",
        ),
    ),
}


class LeaderboardFetcher:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.logs: List[str] = []

    def build(self) -> Dict[str, Any]:
        trending = self._fetch_trending_models()
        leaders: List[Dict[str, Any]] = []
        for entry in trending:
            repo_id = entry.get("modelId") or entry.get("id")
            if not repo_id:
                continue
            scores = self._collect_scores(repo_id)
            if scores["scores"]:
                leaders.append(scores)
        return self._compose_tables(leaders)

    def log_text(self) -> str:
        if not self.logs:
            return "No actions recorded."
        return "\n".join(self.logs)

    def _fetch_trending_models(self) -> List[Dict[str, Any]]:
        params = {"sort": "trendingScore", "limit": TRENDING_FETCH_LIMIT}
        response = self.session.get(
            f"{API_BASE}/models",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected trending response.")
        filtered = [
            model
            for model in data
            if (model.get("pipeline_tag") == PIPELINE_FILTER or PIPELINE_FILTER in (model.get("tags") or []))
        ]
        if not filtered:
            self.logs.append("⚠️ No text-generation models in trending feed.")
            return []
        limited = filtered[:TRENDING_LIMIT]
        if len(limited) < TRENDING_LIMIT:
            self.logs.append(f"⚠️ Only {len(limited)} text-generation models available.")
        else:
            self.logs.append(f"🔍 Loaded {TRENDING_LIMIT} trending text-generation models.")
        return limited

    def _collect_scores(self, repo_id: str) -> Dict[str, Any]:
        owner = repo_id.split("/")[0]
        card_meta = self._read_model_card(repo_id)
        model_index = card_meta.get("model-index")
        if model_index:
            self.logs.append(f"✅ {repo_id}: model card metadata found.")
            scores = self._extract_scores(
                repo_id=repo_id,
                model_index=model_index,
                contributor=owner,
                source_type="model-card",
                source_url=f"https://huggingface.co/{repo_id}",
                revision="main",
            )
            if scores:
                return {"model_id": repo_id, "scores": scores}

        prs = self._fetch_pull_requests(repo_id)
        for pr in prs:
            revision = f"refs/pr/{pr['num']}"
            pr_meta = self._read_model_card(repo_id, revision=revision)
            pr_index = pr_meta.get("model-index")
            if not pr_index:
                continue
            author_info = pr.get("author", {}) or {}
            contributor = author_info.get("name") or author_info.get("fullname") or "unknown-author"
            discussion_path = f"{repo_id}/discussions/{pr['num']}"
            source_url = f"https://huggingface.co/{discussion_path}"
            scores = self._extract_scores(
                repo_id=repo_id,
                model_index=pr_index,
                contributor=contributor,
                source_type="pull-request",
                source_url=source_url,
                revision=revision,
            )
            if scores:
                note = f"📝 {repo_id}: PR #{pr['num']} by {contributor}."
                self.logs.append(note)
                return {"model_id": repo_id, "scores": scores}

        self.logs.append(f"⚠️ {repo_id}: no target benchmarks located.")
        return {"model_id": repo_id, "scores": {}}

    def _read_model_card(
        self,
        repo_id: str,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                revision=revision,
            )
        except HfHubHTTPError as err:
            ctx = f"{repo_id} ({revision or 'main'})"
            self.logs.append(f"🚫 {ctx}: README download failed ({err}).")
            return {}
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self._parse_front_matter(text)

    @staticmethod
    def _parse_front_matter(content: str) -> Dict[str, Any]:
        content = content.lstrip("\ufeff")
        if not content.startswith("---"):
            return {}
        lines = content.splitlines()
        end_idx = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            return {}
        front_matter = "\n".join(lines[1:end_idx])
        try:
            data = yaml.safe_load(front_matter) or {}
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}

    def _fetch_pull_requests(self, repo_id: str) -> List[Dict[str, Any]]:
        url = f"{API_BASE}/models/{repo_id}/discussions"
        try:
            response = self.session.get(
                url,
                params={"limit": PR_SCAN_LIMIT},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as err:
            self.logs.append(f"🚫 {repo_id}: PR list request failed ({err}).")
            return []

        payload = response.json()
        discussions = payload.get("discussions", [])
        prs = [disc for disc in discussions if disc.get("isPullRequest")]
        prs.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
        if prs:
            self.logs.append(f"📬 {repo_id}: scanning {len(prs)} pull requests.")
        return prs

    def _extract_scores(
        self,
        repo_id: str,
        model_index: Any,
        contributor: str,
        source_type: str,
        source_url: str,
        revision: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(model_index, list):
            return {}
        scores: Dict[str, Dict[str, Any]] = {}
        for entry in model_index:
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("name") or repo_id.split("/")[-1]
            for result in entry.get("results", []):
                dataset_info = result.get("dataset") or {}
                dataset_name = dataset_info.get("name")
                dataset_type = dataset_info.get("type")
                task_info = result.get("task") or {}
                task_type = task_info.get("type")
                for metric in result.get("metrics", []):
                    benchmark_key = self._match_benchmark(
                        dataset_name,
                        dataset_type,
                        metric,
                    )
                    if not benchmark_key:
                        continue
                    raw_value = metric.get("value")
                    value = _coerce_score(raw_value)
                    if value is None:
                        continue
                    unit = metric.get("unit") or ""
                    is_pct = isinstance(raw_value, str) and raw_value.strip().endswith("%")
                    if not unit and is_pct:
                        unit = "%"
                    metric_name = metric.get("name") or metric.get("type") or ""
                    payload = {
                        "model": repo_id,
                        "model_name": model_name,
                        "benchmark_key": benchmark_key,
                        "benchmark_label": BENCHMARKS[benchmark_key].label,
                        "value": value,
                        "unit": unit,
                        "dataset": dataset_name or dataset_type or "",
                        "task_type": task_type or "",
                        "metric_name": metric_name,
                        "contributor": contributor,
                        "source_type": source_type,
                        "source_url": source_url,
                        "revision": revision,
                    }
                    existing = scores.get(benchmark_key)
                    if not existing or value > existing["value"]:
                        scores[benchmark_key] = payload
        return scores

    def _match_benchmark(
        self,
        dataset_name: Optional[str],
        dataset_type: Optional[str],
        metric: Dict[str, Any],
    ) -> Optional[str]:
        fields = [
            _normalize(dataset_name),
            _normalize(dataset_type),
            _normalize(metric.get("name")),
            _normalize(metric.get("type")),
        ]
        fields = [field for field in fields if field]
        for key, spec in BENCHMARKS.items():
            if spec.matches(fields):
                return key
        return None

    def _compose_tables(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_rows: List[Dict[str, Any]] = []
        per_benchmark: Dict[str, List[Dict[str, Any]]] = {key: [] for key in BENCHMARKS}
        for entry in entries:
            for benchmark_key, payload in entry["scores"].items():
                row = {
                    "Model": entry["model_id"],
                    "Benchmark": BENCHMARKS[benchmark_key].label,
                    "Score": round(payload["value"], 2),
                    "Source": f"{payload['source_type']} by [{payload['contributor']}]({payload['source_url']})",
                }
                all_rows.append(row)
                per_benchmark[benchmark_key].append(row)

        for rows in per_benchmark.values():
            rows.sort(key=lambda r: r["Score"], reverse=True)
        all_rows.sort(key=lambda r: r["Score"], reverse=True)

        return {
            "all_rows": all_rows,
            "per_benchmark": per_benchmark,
            "stats": {
                "models_with_scores": len(entries),
                "row_count": len(all_rows),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }


def _rows_to_matrix(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    return [[row.get(header, "") for header in TABLE_HEADERS] for row in rows]


def refresh_handler() -> List[Any]:
    fetcher = LeaderboardFetcher()
    try:
        result = fetcher.build()
        stats = result["stats"]
        status = "\n".join(
            [
                f"Last updated: {stats['generated_at']}",
                f"Models with scores: {stats['models_with_scores']}",
                f"Total entries: {stats['row_count']}",
                "",
                fetcher.log_text(),
            ]
        )
        return [
            status,
            _rows_to_matrix(result["all_rows"]),
        ]
    except Exception as exc:  # pylint: disable=broad-except
        error = f"❌ Failed to refresh leaderboard: {exc}"
        empty: List[List[Any]] = []
        return [error, empty]


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # HF Evaluation Leaderboard
        Shows MMLU, BigCodeBench, and ARC MC scores pulled from model-index
        metadata or their pull requests for the top text-generation models.
        """
    )
    refresh_button = gr.Button("Refresh", variant="primary")
    status_box = gr.Markdown("")

    all_table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, datatype=TABLE_DATATYPES)

    refresh_button.click(  # pylint: disable=no-member
        refresh_handler,
        inputs=[],
        outputs=[
            status_box,
            all_table,
        ],
    )
    demo.load(  # pylint: disable=no-member
        refresh_handler,
        outputs=[
            status_box,
            all_table,
        ],
    )


if __name__ == "__main__":
    demo.launch()

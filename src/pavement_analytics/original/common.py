"""Shared I/O, diagnostics, and restartable stage execution (no service imports)."""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from ..security import Boundary, reject_links, source_root

ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = {"SectionUniqueID", "_source_file", "_source_file_start", "_source_file_end",
              "partition_airfield", "partition_section", "partition_original_airfield"}
IDENTITIES |= {name + suffix for name in ("NetworkID", "BranchID", "SectionID", "group_id")
               for suffix in ("", "_start", "_end")}


def read_table(path):
    path = Path(path)
    reject_links(path)
    def check_header(columns):
        columns = [str(c).strip() for c in columns]
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate column names in {path}")
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        with pd.ExcelFile(path) as book:
            header = pd.read_excel(book, header=None, nrows=1)
            if not header.empty:
                check_header(header.iloc[0])
            df = pd.read_excel(book, dtype={c: "string" for c in IDENTITIES},
                               keep_default_na=False, na_values=[""])
    else:
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with path.open(encoding=encoding, newline="") as stream:
                    check_header(next(csv.reader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","), []))
                df = pd.read_csv(path, encoding=encoding, low_memory=False,
                                 sep="\t" if path.suffix.lower() == ".tsv" else ",",
                                 dtype={c: "string" for c in IDENTITIES}, keep_default_na=False, na_values=[""])
                break
            except UnicodeDecodeError:
                continue
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate column names after trimming in {path}")
    return df.replace(r"^\s*$", np.nan, regex=True)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.isoformat()
    return value


def write_json(path, value):
    path = Path(path)
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    reject_links(tmp)
    tmp.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, df):
    path = Path(path)
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    reject_links(tmp)
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256(path):
    reject_links(Path(path))
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_stats(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s[np.isfinite(s)]
    return clean_json({"count": len(valid), "missing_or_nonfinite": len(s) - len(valid),
                       "distinct": valid.nunique(), "zero_pct": 100 * valid.eq(0).mean(),
                       "min": valid.min(), "p10": valid.quantile(.1),
                       "median": valid.median(), "mean": valid.mean(),
                       "p90": valid.quantile(.9), "max": valid.max()})


class Run:
    """Checkpoints are trusted only when inputs, code, config, and outputs match."""
    def __init__(self, output, fingerprint, config, inputs, resume=False):
        self.boundary = Boundary.create(source_root(), Path(output))
        self.output = self.boundary.workspace
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "run_report.json"
        if resume:
            if not self.path.is_file():
                raise ValueError("--resume requires an existing run_report.json")
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("fingerprint") != fingerprint:
                raise ValueError("Inputs, code, or settings changed. Choose a new output directory; this checkpoint cannot be reused.")
            for key, step in self.data["stages"].items():
                if step["status"] in {"completed", "skipped"}:
                    for item in step["artifacts"]:
                        path = self.boundary.path(item["path"])
                        if not path.is_file() or sha256(path) != item["sha256"]:
                            raise ValueError(f"Checkpoint artifact changed or missing: {path}. Choose a new output directory.")
        else:
            if any(self.output.iterdir()):
                raise ValueError("Output directory is not empty. Use --resume for the same run or choose a new directory.")
            self.data = {"schema_version": 1, "fingerprint": fingerprint,
                         "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "config": config, "inputs": inputs, "stages": {}, "warnings": []}
        self.data["status"] = "running"
        self.data.pop("error", None)
        self.data.pop("finished_at", None)
        self.current = "setup"
        self.log = None
        self.save()

    def warn(self, code, message, count=None):
        record = {"stage": self.current, "code": code, "message": message}
        if count is not None:
            record["count"] = int(count)
        self.data["warnings"].append(record)
        print(f"WARNING [{code}] {message}", file=sys.stdout, flush=True)

    def save(self):
        summary = {}
        stages = self.data["stages"]
        for surface in self.data["config"]["surfaces"]:
            model = stages.get(f"07_modeling_{surface}", {}).get("stats", {})
            agreement = stages.get(f"06_pci_{surface}", {}).get("stats", {}).get("end", {})
            summary[surface] = {
                "eligible_inspections": stages.get("01_split", {}).get("stats", {}).get("surfaces", {}).get(surface),
                "pairs": stages.get(f"02_pairs_{surface}", {}).get("stats", {}).get("total_pairs"),
                "model_rows": model.get("rows"), "features": model.get("features"),
                "airfield_groups": model.get("airfield_groups"), "pci_end_mae": agreement.get("mae"),
            }
        self.data["summary"] = clean_json(summary)
        write_json(self.path, self.data)
        warnings_df = pd.DataFrame(self.data["warnings"], columns=["stage", "code", "count", "message"])
        write_csv(self.output / "warnings.csv", warnings_df)
        self.write_markdown()

    def write_markdown(self):
        lines = ["# Curation run report", "", f"Status: **{self.data['status']}**", "",
                 f"Warnings: {len(self.data['warnings'])}", ""]
        if self.data.get("error"):
            lines += [f"Error: {self.data['error']}", ""]
        lines += ["| Surface | Eligible inspections | Pairs | Model rows | Features | Airfield groups | PCI end MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for surface, metrics in self.data["summary"].items():
            values = [f"{v:.3f}" if isinstance(v, float) else str(v) if v is not None else "—" for v in metrics.values()]
            lines.append("| " + " | ".join([surface, *values]) + " |")
        lines += [""]
        lines += ["| Stage | Status | Seconds |", "|---|---|---:|"]
        for key, step in self.data["stages"].items():
            lines.append(f"| {key} | {step['status']} | {step.get('seconds', 0):.2f} |")
        for key, step in self.data["stages"].items():
            lines += ["", f"## {key}", "", "```json", json.dumps(clean_json(step.get("stats", {})), indent=2), "```"]
        if self.data["warnings"]:
            lines += ["", "## Warnings", ""]
            for w in self.data["warnings"]:
                lines.append(f"- **{w['code']}** ({w['stage']}): {w['message']}")
        lines += ["", "Detailed tables, feature decisions, rejected rows, and stage logs are saved alongside this report.", ""]
        (self.output / "run_report.md").write_text("\n".join(lines), encoding="utf-8")

    def stage(self, key, function):
        self.current = key
        old = self.data["stages"].get(key, {})
        if old.get("status") in {"completed", "skipped"}:
            print(f"REUSE {key}", flush=True)
            return old["stats"]
        self.data["warnings"] = [w for w in self.data["warnings"] if w["stage"] != key]
        log_path = self.output / "logs" / f"{key}.log"
        log_path.parent.mkdir(exist_ok=True)
        step = self.data["stages"][key] = {"status": "running", "artifacts": [], "stats": {}}
        self.save()
        print(f"START {key}", flush=True)
        started = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                self.log = log
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        stats, paths = function()
                    except BaseException:
                        traceback.print_exc()
                        raise
            self.log = None
            step.update(status="skipped" if stats.get("skipped") else "completed", stats=clean_json(stats),
                        artifacts=[{"path": str(Path(p).relative_to(self.output)), "sha256": sha256(p)} for p in paths])
        except BaseException as exc:
            self.log = None
            step.update(status="failed", error=str(exc))
            raise
        finally:
            step["seconds"] = round(time.monotonic() - started, 3)
            self.save()
        print(f"DONE  {key} ({step['seconds']:.1f}s)", flush=True)
        return step["stats"]

    def process(self, argv):
        """Use argv directly: paths with spaces work, no shell interpretation."""
        print("Command:", repr([str(a) for a in argv]), flush=True)
        process = subprocess.Popen([str(a) for a in argv], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        notices = []
        try:
            for line in process.stdout:
                self.log.write(line)
                self.log.flush()
                if re.search(r"\[(WARN|ERROR)\]|^\s*!|warning:", line, re.I):
                    notices.append(line.strip())
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if notices:
            self.warn("STAGE_NOTICES", f"Subprocess reported {len(notices)} warnings/errors. See logs/{self.current}.log. First: {notices[0]}", len(notices))
        if code:
            raise RuntimeError(f"{Path(argv[1]).name} exited with code {code}; see logs/{self.current}.log")

    def finish(self, status, error=None):
        self.data.update(status=status, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        if error:
            self.data["error"] = str(error)
        self.save()
        print(f"\n{status.upper()}: {len(self.data['warnings'])} warning(s). Report: {self.output / 'run_report.md'}", flush=True)

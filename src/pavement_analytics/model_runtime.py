"""Single-worker model operations, hard process deadlines and private timing."""
from __future__ import annotations

import contextlib
import contextvars
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import time
import traceback

import pandas as pd

from .io import write_json
from .security import SafeError


ACTIVE = contextvars.ContextVar("pavement_model_runtime", default=None)
DEFAULT_OPTIONS = {"comparisons": True, "isolated": True, "fit_timeout_seconds": 300.,
    "prediction_timeout_seconds": 300., "max_fit_calls": None, "candidate_ids": None,
    "cache_dir": None}
TIMING_COLUMNS = ["candidate", "family", "feature_set", "phase", "fold", "operation",
                  "status", "wall_seconds", "worker_cpu_seconds"]


def sanitize_model_event(event):
    if not isinstance(event, dict) or event.get("event") != "model" or not isinstance(event.get("status"), str) or event.get("status") not in {
            "budget", "started", "completed", "failed", "timed_out", "cancelled", "task_reused"}:
        return None
    result = {"event": "model", "status": event["status"]}
    for key in ("candidate", "family"):
        value = event.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,79}", value):
            result[key] = value
    for key, allowed in (("feature_set", {"all", "nonclimate"}),
        ("phase", {"cv", "primary_refit", "two_stage_refit", "primary_importance", "comparison_refit", "family_importance", "qualification"}),
        ("operation", {"fit", "predict"})):
        if isinstance(event.get(key), str) and event.get(key) in allowed:
            result[key] = event[key]
    for key in ("completed_fits", "planned_fits", "fold"):
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and -1 <= value <= 10**9:
            result[key] = value
    return result


def model_progress_text(event):
    event = sanitize_model_event(event)
    if event is None:
        return None
    if event["status"] == "task_reused":
        return "Reused verified completed model task"
    if event["status"] == "budget":
        return f"Model fit-call upper bound: {event.get('planned_fits', 0):,}"
    return (f"Model {event.get('candidate', '')} / {event.get('feature_set', '')} / "
            f"{event.get('phase', '')} / fold {event.get('fold', -1)}: "
            f"{event.get('operation', '')} {event['status']} "
            f"(fit calls {event.get('completed_fits', 0):,}; upper bound {event.get('planned_fits', 0):,})")


def model_options(raw=None):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw)-set(DEFAULT_OPTIONS):
        raise SafeError("INVALID_LOCAL_MODEL_OPTIONS")
    options = {**DEFAULT_OPTIONS, **raw}
    if any(not isinstance(options[k], bool) for k in ("comparisons", "isolated")):
        raise SafeError("INVALID_LOCAL_MODEL_OPTIONS")
    for key in ("fit_timeout_seconds", "prediction_timeout_seconds"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SafeError("INVALID_LOCAL_MODEL_OPTIONS")
    value = options["max_fit_calls"]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise SafeError("INVALID_LOCAL_MODEL_OPTIONS")
    if options["cache_dir"] is not None and (not isinstance(options["cache_dir"], str) or not options["cache_dir"].strip()):
        raise SafeError("INVALID_LOCAL_MODEL_OPTIONS")
    ids = options["candidate_ids"]
    if ids is not None:
        from .model_registry import candidate_catalog
        known = {x["candidate"] for x in candidate_catalog()}-{"median"}
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= known):
            raise SafeError("UNKNOWN_OR_DUPLICATE_MODEL_CANDIDATE")
    return options


def fit_budget(folds, options, *, constant=False, feature_sets=2):
    from .model_registry import candidate_catalog
    catalog = [x for x in candidate_catalog() if x["candidate"] == "median" or
               (not constant and (options["candidate_ids"] is None or x["candidate"] in options["candidate_ids"]))]
    available = [x for x in catalog if x.get("availability", "available") == "available"]
    count = len(available)
    stages = sum(2 if x["family"] == "two_stage" or x["parameters"].get("early_stopping_rounds") else 1 for x in available)
    diagnostic = sum(x["family"] == "two_stage" for x in available)
    # Upper bounds include every possible selection fallback/refit, the
    # R6 diagnostic refits, and an extra constant baseline. Native counts
    # additionally cover stopped probes and the two occurrence/severity fits.
    factor = folds+1+int(options["comparisons"])
    return {"candidate_count": len(catalog), "available_candidates": count, "feature_sets": feature_sets,
            "folds": folds, "fit_calls_upper_bound": feature_sets*(factor*count+diagnostic)+1,
            "native_fits_upper_bound": feature_sets*(factor*stages+2*diagnostic)+1,
            "fit_workers": 1, "estimator_threads": 1, "includes_selection_fallbacks": True}


def check_budget(budget, options):
    if options["max_fit_calls"] is not None and budget["fit_calls_upper_bound"] > options["max_fit_calls"]:
        raise SafeError("LOCAL_MODEL_FIT_BUDGET_EXCEEDED")


class ModelTimeout(SafeError):
    pass


def runtime_context(**fields):
    runtime = ACTIVE.get()
    if runtime is not None:
        runtime.context = dict(fields)


def _worker(connection, log_path):
    # Redirect native file descriptors as well as Python streams. Backend
    # diagnostics and tracebacks never enter progress messages or the console.
    with open(log_path, "a", encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from threadpoolctl import threadpool_limits
            from .study_modeling import _fit, _prediction_bundle
            with threadpool_limits(limits=1):
                connection.send(("ready",))
                model, token = None, 0
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        return
                    if request[0] == "close":
                        return
                    operation, payload = request
                    started = time.process_time()
                    try:
                        if operation == "fit":
                            model = _fit(*payload)
                            token += 1
                            result = model
                        else:
                            supplied, X = payload
                            if isinstance(supplied, int):
                                if supplied != token or model is None:
                                    raise SafeError("INVALID_MODEL_WORKER_HANDLE")
                            else:
                                model = supplied
                                token += 1
                            result = _prediction_bundle(model, X)
                        connection.send(("ok", result, token, time.process_time()-started))
                    except Exception:
                        traceback.print_exc(file=log)
                        log.flush()
                        connection.send(("error", None, 0, time.process_time()-started))


class ModelRuntime:
    def __init__(self, out, options, progress=None):
        self.out, self.options, self.progress = Path(out), options, progress
        self.context, self.records = {}, []
        self.process = self.connection = self.cached_model = None
        self.handle, self.fit_calls, self.planned = 0, 0, 0

    def __enter__(self):
        self.context_token = ACTIVE.set(self)
        return self

    def emit(self, status, operation=None):
        if self.progress is not None:
            self.progress({"event": "model", **self.context, "status": status, "operation": operation,
                           "completed_fits": self.fit_calls, "planned_fits": self.planned})

    def budget(self, value):
        self.planned = value["fit_calls_upper_bound"]
        write_json(self.out/"fit_budget.json", value)
        check_budget(value, self.options)
        self.emit("budget")

    def _receive(self, seconds):
        deadline = time.monotonic()+seconds
        while not self.connection.poll(min(.1, max(0, deadline-time.monotonic()))):
            if not self.process.is_alive():
                raise SafeError("MODEL_WORKER_EXITED")
            if time.monotonic() >= deadline:
                raise ModelTimeout("MODEL_OPERATION_TIMED_OUT")
        return self.connection.recv()

    def _start(self):
        if self.process is not None:
            return
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, str(self.out/"model_worker.log")), daemon=True)
        self.process.start()
        child.close()
        if self._receive(60.) != ("ready",):
            raise SafeError("MODEL_WORKER_START_FAILED")

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(2.)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2.)
            self.connection.close()
        self.process = self.connection = self.cached_model = None
        self.handle = 0

    def call(self, operation, *args):
        started, cpu, status = time.perf_counter(), None, "completed"
        self.emit("started", operation)
        try:
            if self.options["isolated"]:
                self._start()
                payload = args
                if operation == "predict":
                    payload = (self.handle if self.cached_model is args[0] else args[0], args[1])
                self.connection.send((operation, payload))
                reply = self._receive(self.options["fit_timeout_seconds" if operation == "fit" else "prediction_timeout_seconds"])
                _, result, self.handle, cpu = reply
                if reply[0] != "ok":
                    self.cached_model = None
                    raise SafeError("MODEL_WORKER_OPERATION_FAILED")
                self.cached_model = result if operation == "fit" else args[0]
            else:
                # Explicit in-process mode supports debugging and test doubles;
                # hard deadlines are available only in the default isolated mode.
                from .study_modeling import _fit, _prediction_bundle
                token = ACTIVE.set(None)
                cpu_start = time.process_time()
                try:
                    result = (_fit if operation == "fit" else _prediction_bundle)(*args)
                finally:
                    ACTIVE.reset(token)
                    cpu = time.process_time()-cpu_start
            return result
        except ModelTimeout:
            status = "timed_out"
            self.close()
            raise
        except KeyboardInterrupt:
            status = "cancelled"
            self.close()
            raise
        except Exception:
            status = "failed"
            if self.options["isolated"]:
                self.close()
            raise
        finally:
            if operation == "fit":
                self.fit_calls += 1
            self.records.append({**self.context, "operation": operation, "status": status,
                                 "wall_seconds": time.perf_counter()-started, "worker_cpu_seconds": cpu})
            self.emit(status, operation)

    def __exit__(self, kind, error, tb):
        self.close()
        ACTIVE.reset(self.context_token)
        if self.out.is_dir():
            pd.DataFrame(self.records, columns=TIMING_COLUMNS).to_csv(self.out/"model_runtime.csv", index=False)
            write_json(self.out/"model_runtime.json", {"status": "cancelled" if kind is KeyboardInterrupt else
                       "failed" if kind else "completed", "fit_calls": self.fit_calls,
                       "fit_calls_upper_bound": self.planned, "options": self.options,
                       "fit_workers": 1, "estimator_threads": 1 if self.options["isolated"] else None})

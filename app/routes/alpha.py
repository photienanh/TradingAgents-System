"""FastAPI endpoints for AlphaGPT daily signal integration."""

import asyncio
import queue
import threading
import logging

from typing import Any, Dict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse


router = APIRouter(prefix="/api/alpha", tags=["alpha"])

@router.get("/library")
async def alpha_library_list() -> Dict[str, Any]:
    """Trả về toàn bộ alpha trong alpha_library.json, sort by ic_oos desc."""
    from alpha.daily_runner import _load_alpha_library, _safe_float
    alphas = _load_alpha_library()
 
    rows = []
    for a in alphas:
        description = str(a.get("description", ""))
        if description.startswith("GP"):
            description = description.split(': ', 1)[-1]
        rows.append({
            "id":          str(a.get("id", "")),
            "description": description,
            "formula":     str(a.get("formula", "")),
            "ic_oos":      _safe_float(a.get("ic_oos")),
            "sharpe_oos":  _safe_float(a.get("sharpe_oos")),
            "return_oos":  _safe_float(a.get("return_oos")),
            "hypothesis":  str(a.get("hypothesis", "")),
            "source":      str(a.get("source", "")),
        })
 
    rows.sort(key=lambda x: (x["ic_oos"] or 0), reverse=True)
    return {
        "total": len(rows),
        "alphas": rows,
    }

@router.delete("/library/{alpha_id}")
async def alpha_library_delete(alpha_id: str) -> Dict[str, Any]:
    """Xóa alpha khỏi alpha_library.json theo id."""
    from alpha.daily_runner import _load_alpha_library, ALPHA_LIBRARY_PATH
    import json
    alphas = _load_alpha_library()
    original_len = len(alphas)
    alphas = [a for a in alphas if str(a.get("id", "")) != alpha_id]
    if len(alphas) == original_len:
        return JSONResponse({"error": f"Alpha '{alpha_id}' not found"}, status_code=404)
    with open(ALPHA_LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(alphas, f, ensure_ascii=False, indent=2)
    return {"deleted": alpha_id, "remaining": len(alphas)}

@router.post("/refresh")
async def alpha_refresh(force: bool = Query(default=True)):
    """Trigger market data refresh và rebuild alpha signals."""
    from alpha.manager import trigger_if_needed
    result = trigger_if_needed(force=force)
    return result 
 
@router.get("/signals")
async def alpha_signals_all(limit: int = Query(default=0, ge=0)):
    from alpha.manager import get_all_signals, get_status
    signals = get_all_signals(limit=limit if limit > 0 else None)
    status  = get_status()
    rows = sorted(signals.values(), key=lambda x: (x.get("signal_today") or 0), reverse=True)
    return {
        "total":       len(rows),
        "as_of":       status.get("last_run_at"),
        "last_run_day": status.get("last_run_day"),
        "signals":     rows,
    }

# ── Pipeline run with SSE log streaming ─────────────────────────────────
 
class _QueueHandler(logging.Handler):
    """Logging handler that pushes records into a queue for SSE streaming."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
 
    def emit(self, record: logging.LogRecord):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


class _AlphaLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        if name.startswith("alpha"):
            return True
        msg = record.getMessage()
        return msg.startswith("[Alpha") or msg.startswith("[Run]")
 
 
async def _pipeline_sse_generator(idea: str, iterations: int):
    """Run alpha pipeline in thread, stream logs via SSE."""
 
    log_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()
    result_holder: Dict[str, Any] = {}
 
    # Attach queue handler to root logger for capture
    handler = _QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s  %(message)s", "%H:%M:%S"))
    handler.setLevel(logging.INFO)
    handler.addFilter(_AlphaLogFilter())
    root_logger = logging.getLogger()
    prev_root_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
 
    def _run():
        import asyncio as _asyncio
        from alpha.run import run_pipeline
        from pathlib import Path as _Path
 
        data_dir = str(_Path(__file__).resolve().parents[2] / "data" / "market_data")
        try:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
            final = loop.run_until_complete(
                run_pipeline(data_dir=data_dir, idea=idea, max_iterations=iterations)
            )
            sota = final.get("sota_alphas", []) if isinstance(final, dict) else []
            result_holder["sota"] = sota
            result_holder["hypothesis"] = final.get("hypothesis", "") if isinstance(final, dict) else ""
            result_holder["iteration"] = final.get("iteration", 0) if isinstance(final, dict) else 0
        except Exception as exc:
            log_queue.put(f"[ERROR] Pipeline failed: {exc}")
        finally:
            loop.close()
            done_event.set()
 
    thread = threading.Thread(target=_run, daemon=True, name="alpha-pipeline-ui")
    thread.start()
 
    # Stream log lines
    while not done_event.is_set() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.2)
            yield f"data: {line}\n\n"
        except queue.Empty:
            if done_event.is_set():
                break
            yield ": heartbeat\n\n"
        await asyncio.sleep(0)
 
    root_logger.removeHandler(handler)
    root_logger.setLevel(prev_root_level)
 
    # Send final result summary
    sota = result_holder.get("sota", [])
    summary_lines = [f"=== Pipeline hoàn tất: {len(sota)} SOTA alphas ==="]
    for a in sota:
        ic  = a.get("ic_oos")
        sh  = a.get("sharpe_oos")
        ret = a.get("return_oos")
        summary_lines.append(
            f"  [{a.get('id','?')}] {a.get('description','')[:60]}"
        )
        summary_lines.append(
            f"    IC={f'{ic:+.4f}' if ic is not None else 'N/A'}"
            f"  Sharpe={f'{sh:+.3f}' if sh is not None else 'N/A'}"
            f"  Return={f'{ret*100:+.1f}%' if ret is not None else 'N/A'}"
        )
        summary_lines.append(f"    formula: {a.get('formula','')[:80]}")
 
    for line in summary_lines:
        yield f"data: {line}\n\n"
 
    yield "data: __DONE__\n\n"
 
 
@router.get("/pipeline/run")
async def alpha_pipeline_run(
    idea:       str = Query(default="", description="Trading idea. Để trống = auto-generate."),
    iterations: int = Query(default=3, ge=1, le=10, description="Số vòng lặp pipeline"),
):
    """Stream log của alpha pipeline qua SSE."""
    return StreamingResponse(
        _pipeline_sse_generator(idea=idea.strip(), iterations=iterations),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
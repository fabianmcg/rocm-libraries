# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU hardware monitoring via amdsmi, implemented as a context manager.

Polls GPU metrics (clocks and temperature) in a background daemon thread
during a benchmark window. If amdsmi is unavailable, all averaged fields
stay at 0.0 — monitoring degrades to a no-op rather than crashing.

Mirrors the monitoring logic of client/src/HardwareMonitor.cpp while
using Python threading semantics instead of std::thread.

Selected metrics API (logged at DEBUG on entry):
  1. amdsmi_get_gpu_metrics_info — rich per-XCD metrics (preferred).
  2. amdsmi_get_gpu_activity     — fallback; no temperature or clock data.
  3. No-op                       — logged as a WARNING if neither is found.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable, List, Optional, Tuple

_log = logging.getLogger(__name__)

# Cached module reference; set on first successful import.
_amdsmiMod = None
# Initialized once per process; processor handles are stable for the session.
_amdsmiInitialized: bool = False
_amdsmiProcessors: list = []

# (tempEdgeDegC, gpuClkMhz, socClkMhz, memClkMhz)
_Sample = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Module import helpers
# ---------------------------------------------------------------------------


def _importAmdsmi():
    """Return the amdsmi module, or None if not available on this system"""
    global _amdsmiMod
    if _amdsmiMod is not None:
        return _amdsmiMod

    # Try pyamdsmi first (future pip-installable name).
    for name in ("pyamdsmi", "amdsmi"):
        try:
            mod = __import__(name)
            _amdsmiMod = mod
            _log.debug("imported gpu metrics module '%s'", name)
            return mod
        except ImportError:
            pass

    # amdsmi ships as a system package under $ROCM_PATH/share/amd_smi.
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    smi_path = os.path.join(rocm, "share", "amd_smi")
    if os.path.isdir(smi_path) and smi_path not in sys.path:
        sys.path.insert(0, smi_path)
    try:
        import amdsmi as _sys_amdsmi  # noqa: PLC0415
        _amdsmiMod = _sys_amdsmi
        _log.debug("imported gpu metrics module 'amdsmi' via %s", smi_path)
        return _sys_amdsmi
    except ImportError:
        pass

    return None


def _initAmdsmi(mod) -> list:
    """Initialize amdsmi once per process and return processor handles"""
    global _amdsmiInitialized, _amdsmiProcessors
    if _amdsmiInitialized:
        return _amdsmiProcessors
    mod.amdsmi_init()
    _amdsmiInitialized = True
    _amdsmiProcessors = mod.amdsmi_get_processor_handles()
    return _amdsmiProcessors


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _asFloat(val) -> float:
    """Convert an amdsmi metric value to float, treating 'N/A' or None as 0.0"""
    if val is None or val == "N/A":
        return 0.0
    return float(val)


def _gfxClkAvg(metrics: dict) -> float:
    """Return the average GFX clock in MHz across all XCDs, or 0.0"""
    clks = metrics.get("current_gfxclks")
    if isinstance(clks, list):
        valid = [float(v) for v in clks if v != "N/A"]
        return sum(valid) / len(valid) if valid else 0.0
    return _asFloat(metrics.get("current_gfxclk"))


def _edgeTemp(metrics: dict) -> float:
    """Return edge temperature in degrees C, falling back to hotspot"""
    temp = _asFloat(metrics.get("temperature_edge"))
    if temp == 0.0:
        temp = _asFloat(metrics.get("temperature_hotspot"))
    return temp


def _collectFromMetricsInfo(mod, proc) -> Callable[[], _Sample]:
    """Return a closure that samples via amdsmi_get_gpu_metrics_info"""
    _log.info("hardware monitor: using amdsmi_get_gpu_metrics_info")

    def collect() -> _Sample:
        try:
            m = mod.amdsmi_get_gpu_metrics_info(proc)
        except Exception as exc:
            _log.debug("amdsmi_get_gpu_metrics_info failed: %s", exc)
            return (0.0, 0.0, 0.0, 0.0)
        return (
            _edgeTemp(m),
            _gfxClkAvg(m),
            _asFloat(m.get("current_socclk")),
            _asFloat(m.get("current_uclk")),
        )

    return collect


def _collectFromActivity(mod, proc) -> Callable[[], _Sample]:
    """Return a closure that samples via amdsmi_get_gpu_activity.

    amdsmi_get_gpu_activity does not expose temperature or clock speeds;
    all metric fields will remain 0.0 with this fallback.
    """
    _log.info("hardware monitor: using amdsmi_get_gpu_activity (no temp/clock data)")

    def collect() -> _Sample:
        try:
            mod.amdsmi_get_gpu_activity(proc)
        except Exception as exc:
            _log.debug("amdsmi_get_gpu_activity failed: %s", exc)
        return (0.0, 0.0, 0.0, 0.0)

    return collect


def _selectCollectFn(mod, proc) -> Optional[Callable[[], _Sample]]:
    """Select the richest available metrics collection function"""
    if hasattr(mod, "amdsmi_get_gpu_metrics_info"):
        return _collectFromMetricsInfo(mod, proc)
    if hasattr(mod, "amdsmi_get_gpu_activity"):
        return _collectFromActivity(mod, proc)
    return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class HardwareMonitor:
    """Context manager that polls GPU metrics in a daemon thread.

    Accumulated per-sample averages are available as instance attributes
    after __exit__ returns. If amdsmi is absent, __enter__ logs a warning
    and all fields remain 0.0.

    Replicates the monitoring logic of client/src/HardwareMonitor.cpp
    using Python threading semantics.
    """

    avgTempEdge: float
    avgGpuClockMhz: float
    avgSocClockMhz: float
    avgMemClockMhz: float

    def __init__(self, deviceId: int = 0, intervalMs: int = 10) -> None:
        self.deviceId = deviceId
        self.intervalMs = intervalMs
        self.avgTempEdge = 0.0
        self.avgGpuClockMhz = 0.0
        self.avgSocClockMhz = 0.0
        self.avgMemClockMhz = 0.0
        self._stopEvent: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[_Sample] = []

    def __enter__(self) -> "HardwareMonitor":
        mod = _importAmdsmi()
        if mod is None:
            _log.warning("amdsmi/pyamdsmi not available; hardware monitoring is a no-op")
            return self

        try:
            procs = _initAmdsmi(mod)
        except Exception as exc:
            _log.warning("amdsmi_init failed (%s); hardware monitoring is a no-op", exc)
            return self

        if self.deviceId >= len(procs):
            _log.warning(
                "device %d not found (%d devices); hardware monitoring is a no-op",
                self.deviceId,
                len(procs),
            )
            return self

        collectFn = _selectCollectFn(mod, procs[self.deviceId])
        if collectFn is None:
            _log.warning("no supported amdsmi metrics API found; hardware monitoring is a no-op")
            return self

        self._stopEvent.clear()
        self._samples = []
        self._thread = threading.Thread(
            target=self._pollLoop, args=(collectFn,), daemon=True
        )
        self._thread.start()
        return self

    def _pollLoop(self, collectFn: Callable[[], _Sample]) -> None:
        """Daemon thread body: poll at intervalMs until stop is signalled"""
        while not self._stopEvent.is_set():
            self._samples.append(collectFn())
            self._stopEvent.wait(timeout=self.intervalMs / 1000.0)

    def __exit__(self, *_) -> None:
        self._stopEvent.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._computeAverages()

    def _computeAverages(self) -> None:
        """Compute per-field averages from accumulated samples"""
        n = len(self._samples)
        if n == 0:
            return
        self.avgTempEdge = sum(s[0] for s in self._samples) / n
        self.avgGpuClockMhz = sum(s[1] for s in self._samples) / n
        self.avgSocClockMhz = sum(s[2] for s in self._samples) / n
        self.avgMemClockMhz = sum(s[3] for s in self._samples) / n

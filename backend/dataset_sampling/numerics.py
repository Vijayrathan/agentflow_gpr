"""Narrow adapter to the bundled solver's rounding and material implementation."""
from __future__ import annotations

import decimal
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "gprMax"
if str(_ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(_ROOT.resolve()))

from gprMax.utilities import round_value
from gprMax.constants import c as C0


def native_version():
    from gprMax._version import __version__
    return __version__


def cell_index(value: float, dx: float) -> int:
    return round_value(value / dx)


def native_dt(dx: float, mode: str) -> float:
    if not math.isfinite(dx) or dx <= 0 or mode not in ("2D", "3D"):
        raise ValueError("positive finite dx and explicit 2D/3D mode required")
    term = (1 / dx) * (1 / dx)
    # Match native addition order, including its directed decimal rounding.
    inverse_squared = term + term if mode == "2D" else term + term + term
    return round_value(1 / (C0 * math.sqrt(inverse_squared)),
                       decimalplaces=decimal.getcontext().prec - 1)


def time_axis(dt: float, duration: float):
    iterations = math.ceil(duration / dt) + 1
    return iterations, (iterations - 1) * dt


def excitation(cfg, wf):
    from backend.validation_tools_new import peak_frequency, ricker_band_hz
    fp = peak_frequency(wf.waveform_center_freq_hz, cfg.center_freq_is_peak)
    flo, fhi = ricker_band_hz(wf.waveform_center_freq_hz, cfg.center_freq_is_peak)
    design_high = cfg.high_freq_factor * fp
    if cfg.contract_version >= 2 and cfg.high_freq_factor < 2.5:
        raise ValueError("Ricker design spectrum requires high_freq_factor >= 2.5")
    # Native Ricker chi=sqrt(2)/fp. Four chi is also the native pulse check.
    duration = 4 * math.sqrt(2) / fp
    start = wf.source_start_time or 0.0
    stop = wf.source_end_time
    if wf.source_start_time is not None and stop is None:
        raise ValueError("An explicit source_start_time requires source_end_time; native timing is a paired interval")
    if start < 0 or stop is not None and (stop <= start or stop - start < duration):
        raise ValueError(f"Source timing truncates the Ricker pulse; on-duration must cover {duration:.6g} s")
    return {"peak_hz": fp, "useful_band_hz": [flo, fhi],
            "design_band_hz": [flo, design_high], "pulse_duration_s": duration,
            "start_s": start, "stop_s": stop,
            "spectral_policy": "Wang useful-band gate; Ricker tails use native extrapolation, not empirical certification"}

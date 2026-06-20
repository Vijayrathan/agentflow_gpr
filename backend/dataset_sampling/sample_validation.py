"""
Post-antenna validation gate.

Runs after the waveform and antenna stages, once the N layer samples have already
been drawn (and validated per-draw inside the sampler). It applies only the
validations that BECOME POSSIBLE at this point — i.e. that need the waveform
and/or antenna:

  * the Peplinski frequency-validity GATE on the waveform band — the soil model is
    only valid 0.3-1.3 GHz, so failing this invalidates EVERY sample; and
  * antenna configuration sanity (axis, resistance, source-timing pair).

The per-sample layer checks (texture closure / porosity / theta_v cap / Peplinski
calibration) are NOT repeated here: they already ran on every draw in the sampler
(layer_sampler._sample_one_layer), which rejects infeasible draws and surfaces the
non-blocking warnings. The global-grid checks (TIER 3) are also NOT run here —
they need the derived grid/domain, computed downstream.

Returns a SampleValidationReport; callers decide whether to proceed (no errors)
or stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
)
from backend.validation_tools_new import (
    validate_waveform_and_peplinski_gate,
    validate_antenna_config,
)


@dataclass
class SampleValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    num_samples: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_waveform_antenna(
    dataset_config: DatasetConfig,
    waveform: ExtractedWaveform,
    antenna: ExtractedAntenna,
) -> SampleValidationReport:
    """Run the waveform Peplinski gate and antenna checks that gate the dataset.

    These are dataset-wide (they depend on the waveform/antenna, not on individual
    draws), so the N samples are validated implicitly: a failing band invalidates
    every sample.
    """
    report = SampleValidationReport(num_samples=dataset_config.num_samples)

    # Peplinski frequency gate — the waveform band must lie inside the model's
    # 0.3-1.3 GHz validity window. This gates the WHOLE dataset.
    e, w = validate_waveform_and_peplinski_gate(
        kind=waveform.waveform_kind or "ricker",
        center_freq_hz=waveform.waveform_center_freq_hz,
        center_is_peak=dataset_config.center_freq_is_peak,
        amplitude=waveform.waveform_amplitude,
    )
    report.errors += [f"[waveform] {m}" for m in e]
    report.warnings += [f"[waveform] {m}" for m in w]

    # Antenna configuration sanity. Source timing lives on the waveform.
    e, w = validate_antenna_config(
        kind=antenna.antenna_kind or "hertzian_dipole",
        axis=antenna.antenna_axis or "x",
        resistance=antenna.resistance,
        source_start_time=waveform.source_start_time,
        source_end_time=waveform.source_end_time,
    )
    report.errors += [f"[antenna] {m}" for m in e]
    report.warnings += [f"[antenna] {m}" for m in w]

    return report

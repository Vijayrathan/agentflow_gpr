"""Prepare and check the two collector prompts without running simulations."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "backend", ROOT / "gprMax"):
    sys.path.insert(0, str(path))

from backend.schema import DatasetConfig, ExtractedLayers, ExtractedWaveform, ExtractedAntenna
from dataset_sampling.layer_sampler import sample_layers
from dataset_sampling.peplinski_derive import derive_samples
from dataset_sampling.global_derive import derive_global
from dataset_sampling.global_validation import validate_global
from dataset_sampling.sample_validation import validate_waveform_antenna
from dataset_sampling.emit import build_in_text

OUT = Path(__file__).resolve().parent


def layer(name, lo, hi, thickness=None):
    result = dict(name=name, sand_pct_min=40, sand_pct_max=40,
                  clay_pct_min=15, clay_pct_max=15, theta_v_min=lo, theta_v_max=hi,
                  bulk_density_gcm3_min=1.5, bulk_density_gcm3_max=1.5,
                  particle_density_gcm3_min=2.65, particle_density_gcm3_max=2.65)
    if thickness:
        result.update(thickness_m_min=thickness[0], thickness_m_max=thickness[1])
    return result


def line(values):
    def value(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)
    return "; ".join(f"{k}={value(v)}" for k, v in values.items()) + "."


results = {}
grids = {}
draws = {}
pending = {}
common_headers = []
for arm, thickness in (("A", (0.18, 0.24)), ("B", (0.21, 0.21))):
    settings = dict(num_samples=100, model_basename=f"moisture_{arm}_pilot",
                    moisture_sampling="uniform_per_sample", moisture_seed=42,
                    dimensionality="2D", pml_cells=10, buffer_cells=10,
                    cells_per_wavelength=10, high_freq_factor=3.0,
                    center_freq_is_peak=True, fractal_nbins=50)
    cfg = DatasetConfig(**settings)
    layer_values = [layer("top_loam", 0.10, 0.30, thickness),
                    layer("middle_loam", 0.05, 0.05, (0.32, 0.32)),
                    layer("lower_loam", 0.08, 0.08)]
    layers = ExtractedLayers(num_layers=3, soil_depth_m=2.0, layers=layer_values)
    waveform = dict(waveform_kind="ricker", waveform_amplitude=1.0,
                    waveform_center_freq_hz=750000000, waveform_name="ricker_750mhz_peak")
    antenna = dict(antenna_kind="hertzian_dipole", antenna_axis="z",
                   tx_rx_offset_m=0.08, resistance=None, source_height_m=0.45,
                   rx_same_height=True)
    wf, ant = ExtractedWaveform(**waveform), ExtractedAntenna(**antenna)
    samples, warnings = sample_layers(layers, cfg.num_samples, seed=42,
                                     moisture_sampling=cfg.moisture_sampling,
                                     moisture_seed=cfg.moisture_seed)
    _, eps = derive_samples(samples, cfg, wf)
    grid = derive_global(cfg, wf, ant, layers, eps.eps_r_max, eps.eps_r_min)
    reports = [validate_waveform_antenna(cfg, wf, ant), validate_global(grid, cfg, wf, ant, layers)]
    errors = [e for report in reports for e in report.errors]
    if errors:
        raise RuntimeError(f"Arm {arm}: {errors}")
    requested_thickness = []
    effective_thickness = []
    for sample in samples:
        deck, labels = build_in_text(sample, grid, cfg, wf, ant, adv=None)
        material_lines = [s for s in deck.splitlines() if s.startswith("#soil_peplinski:")]
        for material, sampled in zip(material_lines, sample.layers, strict=True):
            assert tuple(map(float, material.split()[5:7])) == (sampled.theta_v_min, sampled.theta_v_max)
            assert sampled.theta_v_min == sampled.theta_v_max
        common_headers.append([s for s in deck.splitlines() if s.startswith(
            ("#domain:", "#dx_dy_dz:", "#time_window:", "#pml_cells:", "#waveform:", "#hertzian_dipole:", "#rx:"))])
        requested_thickness.append(sample.layers[0].thickness_m)
        effective_thickness.append(labels[0].thickness_m)
    draws[arm] = [[s.sample_id, *[l.theta_v_min for l in s.layers]] for s in samples]
    grids[arm] = grid.model_dump()
    results[arm] = dict(samples_checked=len(samples), errors=errors,
                        warnings=warnings + [w for r in reports for w in r.warnings],
                        top_thickness_requested_min=min(requested_thickness),
                        top_thickness_requested_max=max(requested_thickness),
                        top_thickness_effective_min=min(effective_thickness),
                        top_thickness_effective_max=max(effective_thickness))
    text = "Generate a pilot dataset for first-layer volumetric moisture estimation.\n\n"
    text += "Dataset configuration:\n" + line(settings) + "\n\n"
    text += "Please define all 3 layers now:\nsoil_depth_m=2.0.\n"
    text += "\n".join(f"Layer {i}: {line(values)}" for i, values in enumerate(layer_values, 1))
    text += "\nLayer 3 is the terminal half-space; omit its thickness fields. It fills the remaining depth.\n"
    text += "\nBuried targets:\nExplicitly skip all buried targets. Save target_ranges as {}.\n"
    text += "\nWaveform:\n" + line(waveform)
    text += "\n\nAntenna:\n" + line(antenna)
    text += "\n\nAdvanced parameters:\nExplicitly skip surface roughness and snapshots. Save advanced_params as {}. No receiver array.\n"
    text += "\nUse the supplied values exactly. Keep texture, densities, lower-layer moisture, acquisition and numerical settings fixed. "
    text += "Draw first-layer moisture independently of thickness; emit each sampled moisture as equal native minimum and maximum bounds. "
    text += "The moisture range is a population range, not spatial moisture variation within a layer. "
    text += "Generate the input files; do not start simulations automatically. If validation fails, report the failure without changing the scene.\n"
    pending[arm] = text

assert draws["A"] == draws["B"]
assert grids["A"] == grids["B"]
assert all(header == common_headers[0] for header in common_headers)
for arm, text in pending.items():
    (OUT / f"INPUT_{arm}.txt").write_text(text)
summary = dict(stage="scene template generation checks; no field solves",
               arms=results, exact_moisture_pairing=True, identical_global_plan=True,
               identical_emitted_acquisition_and_numerics=True, global_plan=grids["A"])
(OUT / "SCENE_TEMPLATE_CHECKS.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

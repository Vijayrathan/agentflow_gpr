"""
Batch GPRMax .in file generation from resolved layer ranges.
Called by simulate_workflow() when all ranges + num_samples are collected.
"""
import csv
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

from schema import (
    LayerSchema,
    GprSchema,
    SampledLayerValues,
    SampleRecord,
    DatasetGenerationResult,
)
from physics_modelling import (
    generate_gprmax_input_file,
    SurfaceRoughnessConfig,
    RxArrayConfig,
    SnapshotConfig,
)
from resolvers import ResolvedLayerRange

logger = logging.getLogger(__name__)


def _sample_uniform(lo: float, hi: float, rng: random.Random) -> float:
    """Sample uniformly from [lo, hi]; returns lo if lo == hi."""
    return rng.uniform(lo, hi) if lo < hi else lo


def _sample_texture(
    r: ResolvedLayerRange,
    rng: random.Random,
    max_retries: int = 200,
) -> Tuple[float, float, float]:
    """Accept-reject sampling for sand/silt/clay that sums to 100.

    Draw raw values independently, normalise to 100, accept if all three
    normalised values fall within the user-stated ranges.
    """
    for _ in range(max_retries):
        sr = rng.uniform(r.sand_pct_min, r.sand_pct_max)
        si = rng.uniform(r.silt_pct_min, r.silt_pct_max)
        cr = rng.uniform(r.clay_pct_min, r.clay_pct_max)
        total = sr + si + cr
        if total <= 0:
            continue
        s = 100 * sr / total
        si2 = 100 * si / total
        c = 100 * cr / total
        if (
            r.sand_pct_min <= s <= r.sand_pct_max
            and r.silt_pct_min <= si2 <= r.silt_pct_max
            and r.clay_pct_min <= c <= r.clay_pct_max
        ):
            return round(s, 6), round(si2, 6), round(c, 6)
    raise ValueError(
        f"Cannot sample valid sand/silt/clay in the given ranges after {max_retries} retries. "
        "Please relax the ranges so they are wider or less constrained."
    )


def _sample_layer(
    r: ResolvedLayerRange,
    rng: random.Random,
) -> Tuple[SampledLayerValues, LayerSchema]:
    """Sample one concrete layer from a range spec."""
    thickness = _sample_uniform(r.thickness_m_min, r.thickness_m_max, rng)
    sand, silt, clay = _sample_texture(r, rng)
    theta_v = _sample_uniform(r.theta_v_min, r.theta_v_max, rng)

    bd = (
        _sample_uniform(r.bulk_density_gcm3_min, r.bulk_density_gcm3_max, rng)
        if r.bulk_density_gcm3_min is not None
        else None
    )
    pd = (
        _sample_uniform(r.particle_density_gcm3_min, r.particle_density_gcm3_max, rng)
        if r.particle_density_gcm3_min is not None
        else None
    )
    sal = rng.choice(r.salinity_classes) if r.salinity_classes else None

    sv = SampledLayerValues(
        name=r.name,
        thickness_m=thickness,
        sand_pct=sand,
        silt_pct=silt,
        clay_pct=clay,
        theta_v=theta_v,
        bulk_density_gcm3=bd,
        particle_density_gcm3=pd,
        organic_fraction=r.organic_fraction,
        salinity_class=sal,
    )
    ls = LayerSchema(
        name=r.name,
        thickness_m=thickness,
        sand_pct=sand,
        silt_pct=silt,
        clay_pct=clay,
        theta_v=theta_v,
        bulk_density_gcm3=bd,
        particle_density_gcm3=pd,
        organic_fraction=r.organic_fraction,
        salinity_class=sal,
        porewater_sigma_Sm=r.porewater_sigma_Sm,
    )
    return sv, ls


def _gpr_schema_for_sample(
    template: GprSchema,
    sampled_layers: List[LayerSchema],
    title: str,
) -> GprSchema:
    """Clone the template GprSchema with new layers and title."""
    return GprSchema(
        model=template.model,
        title=title,
        source_height_m=template.source_height_m,
        domain_x=template.domain_x,
        domain_y=template.domain_y,
        cells_per_wavelength=template.cells_per_wavelength,
        max_cell_m=template.max_cell_m,
        temperature_c=template.temperature_c,
        enforce_validity=template.enforce_validity,
        waveform=template.waveform,
        antenna=template.antenna,
        layers=sampled_layers,
        surface_roughness=template.surface_roughness,
        snapshots=template.snapshots,
        rx_array=template.rx_array,
        pml_cells=template.pml_cells,
        num_threads=template.num_threads,
        output_dir=None,  # output_dir is managed by dataset_generator
    )


def _call_generate(gpr: GprSchema, output_filepath: str) -> None:
    """Call physics_modelling.generate_gprmax_input_file with the given schema."""
    layer_thicknesses_m = [l.thickness_m for l in gpr.layers]
    layer_sand_pcts = [l.sand_pct for l in gpr.layers]
    layer_silt_pcts = [l.silt_pct for l in gpr.layers]
    layer_clay_pcts = [l.clay_pct for l in gpr.layers]
    layer_theta_vs = [l.theta_v for l in gpr.layers]
    layer_bulk_densities_gcm3 = [l.bulk_density_gcm3 for l in gpr.layers]
    layer_particle_densities_gcm3 = [l.particle_density_gcm3 for l in gpr.layers]
    layer_organic_fractions = [
        l.organic_fraction if l.organic_fraction is not None else 0.0
        for l in gpr.layers
    ]
    layer_salinity_classes = [l.salinity_class for l in gpr.layers]
    layer_porewater_sigmas_Sm = [l.porewater_sigma_Sm for l in gpr.layers]
    layer_names = [l.name for l in gpr.layers]

    # Convert surface_roughness schema to dataclass
    surface_roughness = None
    if gpr.surface_roughness is not None:
        sr = gpr.surface_roughness
        surface_roughness = SurfaceRoughnessConfig(
            fractal_dim=sr.fractal_dim,
            weight_x=sr.weight_x,
            weight_y=sr.weight_y,
            amplitude_m=sr.amplitude_m,
            add_water=sr.add_water,
            water_depth_m=sr.water_depth_m,
            seed=sr.seed,
        )

    # Convert rx_array schema to dataclass
    rx_array = None
    if gpr.rx_array is not None:
        ra = gpr.rx_array
        rx_array = RxArrayConfig(
            x1=ra.x1, y1=ra.y1, z1=ra.z1,
            x2=ra.x2, y2=ra.y2, z2=ra.z2,
            dx=ra.dx, dy=ra.dy, dz=ra.dz,
        )

    # Convert snapshots schema to dataclass list
    snapshots = None
    if gpr.snapshots is not None:
        snapshots = []
        for s in gpr.snapshots:
            snapshots.append(SnapshotConfig(
                time_s=s.time_s, filename=s.filename,
                dx=s.dx, dy=s.dy, dz=s.dz,
                x1=s.x1, y1=s.y1, z1=s.z1,
                x2=s.x2, y2=s.y2, z2=s.z2,
            ))

    output_dir = str(Path(output_filepath).parent)
    output_filename = str(output_filepath)

    generate_gprmax_input_file(
        layer_thicknesses_m=layer_thicknesses_m,
        layer_sand_pcts=layer_sand_pcts,
        layer_silt_pcts=layer_silt_pcts,
        layer_clay_pcts=layer_clay_pcts,
        layer_theta_vs=layer_theta_vs,
        layer_bulk_densities_gcm3=layer_bulk_densities_gcm3,
        layer_particle_densities_gcm3=layer_particle_densities_gcm3,
        layer_organic_fractions=layer_organic_fractions,
        layer_salinity_classes=layer_salinity_classes,
        layer_porewater_sigmas_Sm=layer_porewater_sigmas_Sm,
        layer_names=layer_names,
        waveform_kind=gpr.waveform.kind,
        waveform_amplitude=gpr.waveform.amplitude,
        waveform_center_freq_hz=gpr.waveform.center_freq_hz,
        waveform_name=gpr.waveform.name,
        antenna_kind=gpr.antenna.kind,
        antenna_axis=gpr.antenna.axis,
        antenna_tx_rx_offset_m=gpr.antenna.tx_rx_offset_m,
        antenna_source_start_time=gpr.antenna.source_start_time,
        antenna_source_end_time=gpr.antenna.source_end_time,
        model_title=gpr.title,
        source_height_m=gpr.source_height_m,
        domain_xy_m=(gpr.domain_x, gpr.domain_y),
        cells_per_wavelength=int(gpr.cells_per_wavelength),
        max_cell_m=gpr.max_cell_m,
        rx_same_height=True,
        temperature_c=gpr.temperature_c,
        model=gpr.model,
        enforce_validity=gpr.enforce_validity,
        output_filename=output_filename,
        objects=None,
        pml_cells=gpr.pml_cells,
        num_threads=gpr.num_threads,
        output_dir=output_dir,
        surface_roughness=surface_roughness,
        snapshots=snapshots,
        rx_array=rx_array,
    )


def _write_manifest(
    output_dir: Path,
    samples: List[SampleRecord],
    errors: List[str],
) -> Tuple[str, str]:
    """Write manifest.csv and manifest.json. Returns (csv_path, json_path)."""
    csv_path = output_dir / "manifest.csv"
    json_path = output_dir / "manifest.json"

    # Build CSV rows
    if samples:
        # Determine layer column names from first sample
        n_layers = len(samples[0].layers)
        layer_fields = [
            "thickness_m", "sand_pct", "silt_pct", "clay_pct", "theta_v",
            "bulk_density_gcm3", "particle_density_gcm3", "organic_fraction",
            "salinity_class",
        ]
        fieldnames = ["sample_index", "filename", "filepath"]
        for li in range(n_layers):
            for f in layer_fields:
                fieldnames.append(f"layer_{li + 1}_{f}")

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rec in samples:
                row = {
                    "sample_index": rec.sample_index,
                    "filename": rec.filename,
                    "filepath": rec.filepath,
                }
                for li, lv in enumerate(rec.layers):
                    prefix = f"layer_{li + 1}_"
                    row[prefix + "thickness_m"] = lv.thickness_m
                    row[prefix + "sand_pct"] = lv.sand_pct
                    row[prefix + "silt_pct"] = lv.silt_pct
                    row[prefix + "clay_pct"] = lv.clay_pct
                    row[prefix + "theta_v"] = lv.theta_v
                    row[prefix + "bulk_density_gcm3"] = lv.bulk_density_gcm3
                    row[prefix + "particle_density_gcm3"] = lv.particle_density_gcm3
                    row[prefix + "organic_fraction"] = lv.organic_fraction
                    row[prefix + "salinity_class"] = lv.salinity_class
                writer.writerow(row)
    else:
        # Write empty CSV with minimal headers
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["sample_index", "filename", "filepath"])
            writer.writeheader()

    # Write JSON
    manifest_data = {
        "samples": [rec.model_dump() for rec in samples],
        "generation_errors": errors,
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(manifest_data, jf, indent=2, default=str)

    return str(csv_path), str(json_path)


def generate_dataset(
    resolved_layer_ranges: List[ResolvedLayerRange],
    gpr_schema_template: GprSchema,
    num_samples: int,
    dataset_name: str,
    title_prefix: str = "gpr_dataset",
    seed: Optional[int] = None,
    max_retries_per_sample: int = 50,
) -> DatasetGenerationResult:
    """Generate num_samples GPRMax .in files by sampling from layer ranges.

    Args:
        resolved_layer_ranges: List of ResolvedLayerRange (one per layer).
        gpr_schema_template: Fixed GprSchema with antenna/waveform/model params;
            layers are replaced per sample.
        num_samples: How many .in files to generate.
        dataset_name: Name used for the output directory.
        title_prefix: Prefix for each file's title (e.g. "gpr_dataset").
        seed: Optional random seed for reproducibility.
        max_retries_per_sample: Max attempts per sample before marking it failed.

    Returns:
        DatasetGenerationResult with generation summary and manifest paths.
    """
    from generator_agent import get_workspace_directory, validate_gpr_parameters

    rng = random.Random(seed)

    # Set up output directories
    workspace_dir = get_workspace_directory()
    dataset_dir = workspace_dir / "datasets" / dataset_name
    files_dir = dataset_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[DATASET] Generating {num_samples} samples into {dataset_dir}"
    )

    samples: List[SampleRecord] = []
    errors: List[str] = []
    num_generated = 0
    num_failed = 0

    for i in range(1, num_samples + 1):
        sample_title = f"{title_prefix}_{i:04d}"
        output_filepath = str(files_dir / f"{sample_title}.in")

        succeeded = False
        sample_failed = False
        for attempt in range(max_retries_per_sample):
            try:
                # 1. Sample each layer
                sampled_layer_values: List[SampledLayerValues] = []
                sampled_layer_schemas: List[LayerSchema] = []
                for r in resolved_layer_ranges:
                    sv, ls = _sample_layer(r, rng)
                    sampled_layer_values.append(sv)
                    sampled_layer_schemas.append(ls)

                # 2. Build GprSchema for this sample
                gpr_sample = _gpr_schema_for_sample(
                    gpr_schema_template, sampled_layer_schemas, sample_title
                )

                # 3. Validate physics constraints
                is_valid, err_msg = validate_gpr_parameters(gpr_sample)
                if not is_valid:
                    logger.debug(
                        f"[DATASET] Sample {i} attempt {attempt + 1} invalid: {err_msg[:100]}"
                    )
                    continue

                # 4. Generate the .in file
                _call_generate(gpr_sample, output_filepath)

                # 5. Record success
                record = SampleRecord(
                    sample_index=i,
                    filename=f"{sample_title}.in",
                    filepath=output_filepath,
                    layers=sampled_layer_values,
                )
                samples.append(record)
                num_generated += 1
                succeeded = True
                logger.debug(f"[DATASET] Sample {i} generated: {output_filepath}")
                break

            except ValueError as e:
                # Texture sampling or validation error — non-retryable
                err = f"Sample {i}: {e}"
                errors.append(err)
                logger.warning(f"[DATASET] {err}")
                num_failed += 1
                sample_failed = True
                break
            except Exception as e:
                logger.warning(
                    f"[DATASET] Sample {i} attempt {attempt + 1} error: {e}"
                )
                if attempt == max_retries_per_sample - 1:
                    err = f"Sample {i}: exhausted {max_retries_per_sample} retries — {e}"
                    errors.append(err)
                    num_failed += 1
                    sample_failed = True

        if not succeeded and not sample_failed:
            # Exhausted retries due to repeated validation failures
            err = f"Sample {i}: exhausted {max_retries_per_sample} retries (all samples failed physics validation)"
            errors.append(err)
            logger.warning(f"[DATASET] {err}")
            num_failed += 1

    # Write manifest
    csv_path, json_path = _write_manifest(dataset_dir, samples, errors)

    logger.info(
        f"[DATASET] Complete: {num_generated}/{num_samples} generated, "
        f"{num_failed} failed. Manifest: {csv_path}"
    )

    if num_generated == 0:
        status = "error"
    elif num_failed > 0:
        status = "partial"
    else:
        status = "complete"

    return DatasetGenerationResult(
        status=status,
        dataset_name=dataset_name,
        output_dir=str(dataset_dir),
        num_requested=num_samples,
        num_generated=num_generated,
        num_failed=num_failed,
        manifest_csv_path=csv_path,
        manifest_json_path=json_path,
        samples=samples,
        errors=errors,
    )

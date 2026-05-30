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

from backend.schema import (
    LayerSchema,
    GprSchema,
    SampledLayerValues,
    SampleRecord,
    DatasetGenerationResult,
)
from backend.physics_modelling import build_gprmax_input
from dataset_sampling.resolvers import ResolvedLayerRange
from dataset_sampling.validation import (
    validate_sampled_layer,
    clamp_texture_to_model,
    validate_texture_feasibility,
    clamp_theta_v_to_model,
)

logger = logging.getLogger(__name__)

# Physics model fallback density values (matching physics_modelling.py defaults).
# Used as display-only defaults in the manifest when the user does not supply ranges,
# so the manifest reflects what was actually used in the dielectric computation
# for Peplinski and Dobson models.
# NOTE: CRIM and Mironov use texture-based porosity estimation when density is None,
#       so the fallback values here are NOT necessarily what those models use.
FALLBACK_BULK_DENSITY_GCM3 = 1.5
FALLBACK_PARTICLE_DENSITY_GCM3 = 2.66


def _sample_uniform(lo: float, hi: float, rng: random.Random) -> float:
    """Sample uniformly from [lo, hi]; returns lo if lo == hi."""
    return rng.uniform(lo, hi) if lo < hi else lo


def _sample_texture(
    r: ResolvedLayerRange,
    rng: random.Random,
    model: str = "crim",
    max_retries: int = 200,
) -> Tuple[float, float, float]:
    """Accept-reject sampling for sand/silt/clay that sums to 100.

    Draw raw values independently, normalise to 100, accept if all three
    normalised values fall within the user-stated ranges (clamped to model
    validity bounds when applicable).
    """
    sand_lo, sand_hi, silt_lo, silt_hi, clay_lo, clay_hi = clamp_texture_to_model(
        r.sand_pct_min, r.sand_pct_max,
        r.silt_pct_min, r.silt_pct_max,
        r.clay_pct_min, r.clay_pct_max,
        model,
    )
    validate_texture_feasibility(
        sand_lo, sand_hi, silt_lo, silt_hi, clay_lo, clay_hi, model,
    )

    for _ in range(max_retries):
        sr = rng.uniform(sand_lo, sand_hi)
        si = rng.uniform(silt_lo, silt_hi)
        cr = rng.uniform(clay_lo, clay_hi)
        total = sr + si + cr
        if total <= 0:
            continue
        s = 100 * sr / total
        si2 = 100 * si / total
        c = 100 * cr / total
        if (
            sand_lo <= s <= sand_hi
            and silt_lo <= si2 <= silt_hi
            and clay_lo <= c <= clay_hi
        ):
            s_r = round(s, 6)
            si_r = round(si2, 6)
            c_r = round(100.0 - s_r - si_r, 6)
            return s_r, si_r, c_r
    raise ValueError(
        f"Cannot sample valid sand/silt/clay in the given ranges after {max_retries} retries. "
        "Please relax the ranges so they are wider or less constrained."
    )


def _sample_layer(
    r: ResolvedLayerRange,
    rng: random.Random,
    model: str = "crim",
) -> Tuple[SampledLayerValues, LayerSchema]:
    """Sample one concrete layer from a range spec and validate it.

    Returns (SampledLayerValues, LayerSchema).  The LayerSchema keeps density as
    None when not provided by the user so physics_modelling.py chooses the correct
    computation path (texture-based porosity for CRIM/Mironov vs. the 1.5/2.66
    fallback for Peplinski/Dobson).  The SampledLayerValues records either the
    user-sampled value or the physics fallback so the manifest is never blank.

    Raises ValueError if the sampled values violate physics constraints
    (density ordering, porosity vs theta_v, model-specific bounds).
    """
    thickness = _sample_uniform(r.thickness_m_min, r.thickness_m_max, rng)
    sand, silt, clay = _sample_texture(r, rng, model=model)

    # Clamp theta_v sampling range to model-specific max
    tv_lo, tv_hi = clamp_theta_v_to_model(r.theta_v_min, r.theta_v_max, model)
    theta_v = _sample_uniform(tv_lo, tv_hi, rng)

    # Actual sampled value (None when not provided — preserves physics model behaviour)
    bd_sampled = (
        _sample_uniform(r.bulk_density_gcm3_min, r.bulk_density_gcm3_max, rng)
        if r.bulk_density_gcm3_min is not None
        else None
    )
    pd_sampled = (
        _sample_uniform(r.particle_density_gcm3_min, r.particle_density_gcm3_max, rng)
        if r.particle_density_gcm3_min is not None
        else None
    )

    # Validate sampled values against physics constraints
    err = validate_sampled_layer(sand, silt, clay, theta_v, bd_sampled, pd_sampled, model)
    if err is not None:
        raise ValueError(err)

    sal = rng.choice(r.salinity_classes) if r.salinity_classes else None

    porosity_sampled = (
        _sample_uniform(r.porosity_min, r.porosity_max, rng)
        if r.porosity_min is not None
        else None
    )

    # Manifest values: use sampled value when provided, fallback constant otherwise.
    # This makes the manifest a complete record of what was used in the computation.
    bd_manifest = bd_sampled if bd_sampled is not None else FALLBACK_BULK_DENSITY_GCM3
    pd_manifest = pd_sampled if pd_sampled is not None else FALLBACK_PARTICLE_DENSITY_GCM3

    sv = SampledLayerValues(
        name=r.name,
        thickness_m=thickness,
        sand_pct=sand,
        silt_pct=silt,
        clay_pct=clay,
        theta_v=theta_v,
        bulk_density_gcm3=bd_manifest,
        particle_density_gcm3=pd_manifest,
        porosity=porosity_sampled,
        organic_fraction=r.organic_fraction,
        salinity_class=sal,
        porewater_sigma_Sm=r.porewater_sigma_Sm,
    )
    ls = LayerSchema(
        name=r.name,
        thickness_m=thickness,
        sand_pct=sand,
        silt_pct=silt,
        clay_pct=clay,
        theta_v=theta_v,
        # Pass the actual sampled value (or None) to physics_modelling so the
        # model selects the appropriate dielectric computation path.
        bulk_density_gcm3=bd_sampled,
        particle_density_gcm3=pd_sampled,
        porosity=porosity_sampled,
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
        domain_xy_m=template.domain_xy_m,
        cells_per_wavelength=template.cells_per_wavelength,
        max_cell_m=template.max_cell_m,
        rx_same_height=template.rx_same_height,
        temperature_c=template.temperature_c,
        enforce_validity=template.enforce_validity,
        waveform=template.waveform,
        antenna=template.antenna,
        layers=sampled_layers,
        objects=template.objects,
        surface_roughness=template.surface_roughness,
        snapshots=template.snapshots,
        rx_array=template.rx_array,
        pml_cells=template.pml_cells,
        num_threads=template.num_threads,
        output_dir=None,  # output_dir is managed by dataset_generator
        fractal_nbins=template.fractal_nbins,
    )


def _call_generate(gpr: GprSchema, output_filepath: str) -> None:
    """Generate a gprMax .in file from a resolved GprSchema."""
    build_gprmax_input(gpr, output_filepath)


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
            "name", "thickness_m", "sand_pct", "silt_pct", "clay_pct",
            "theta_v", "bulk_density_gcm3", "particle_density_gcm3",
            "porosity", "organic_fraction", "salinity_class",
            "porewater_sigma_Sm",
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
                    row[prefix + "name"] = lv.name
                    row[prefix + "thickness_m"] = lv.thickness_m
                    row[prefix + "sand_pct"] = lv.sand_pct
                    row[prefix + "silt_pct"] = lv.silt_pct
                    row[prefix + "clay_pct"] = lv.clay_pct
                    row[prefix + "theta_v"] = lv.theta_v
                    row[prefix + "bulk_density_gcm3"] = lv.bulk_density_gcm3
                    row[prefix + "particle_density_gcm3"] = lv.particle_density_gcm3
                    row[prefix + "porosity"] = lv.porosity
                    row[prefix + "organic_fraction"] = lv.organic_fraction
                    row[prefix + "salinity_class"] = lv.salinity_class
                    row[prefix + "porewater_sigma_Sm"] = lv.porewater_sigma_Sm
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

    rng = random.Random(seed)

    # Validate frequency against model validity band
    from dataset_sampling.validation import validate_frequency_for_model
    freq_err = validate_frequency_for_model(
        gpr_schema_template.waveform.center_freq_hz,
        gpr_schema_template.model,
    )
    if freq_err is not None:
        logger.warning(f"[DATASET] Frequency validation warning: {freq_err}")

    # Set up output directories — always relative to the project root,
    # regardless of the current working directory.
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "datasets" / dataset_name
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
                    sv, ls = _sample_layer(r, rng, model=gpr_schema_template.model)
                    sampled_layer_values.append(sv)
                    sampled_layer_schemas.append(ls)

                # 2. Build GprSchema for this sample
                gpr_sample = _gpr_schema_for_sample(
                    gpr_schema_template, sampled_layer_schemas, sample_title
                )

                # 3. Generate the .in file
                _call_generate(gpr_sample, output_filepath)

                # 4. Record success
                record = SampleRecord(
                    sample_index=i,
                    filename=f"{sample_title}.in",
                    filepath=output_filepath,
                    layers=sampled_layer_values,
                )
                samples.append(record)
                num_generated += 1
                succeeded = True
                logger.info(f"[DATASET] Sample {i} generated: {output_filepath}")
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

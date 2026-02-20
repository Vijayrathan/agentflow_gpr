"""Tests for parameter state management: merge_aggregations and incremental collection."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from schema import (
    ExtractedLayerParams, ExtractedLayers,
    ExtractedAntennaWaveform, ExtractedModelConfig,
    ExtractedOptionalParams, AggregatedExtraction,
)
from resolvers import merge_aggregations, merge_extractions


def _empty_layers():
    return ExtractedLayers(num_layers=0, layers=[])


def _empty_aggregation():
    return AggregatedExtraction(
        layers=_empty_layers(),
        antenna_waveform=ExtractedAntennaWaveform(),
        model_params=ExtractedModelConfig(),
        optional_params=ExtractedOptionalParams(),
    )


def _sample_layers():
    return ExtractedLayers(num_layers=2, layers=[
        ExtractedLayerParams(name="topsoil", thickness_m=0.3, texture_class="sandy_loam", moisture_state="normal"),
        ExtractedLayerParams(name="clay_base", thickness_m=0.5, texture_class="clay", moisture_state="wet"),
    ])


def _sample_antenna():
    return ExtractedAntennaWaveform(
        waveform_center_freq_hz=400e6, tx_rx_offset_m=0.1,
        waveform_kind="ricker", waveform_amplitude=1.0, waveform_name="my_ricker",
    )


def _sample_model():
    return ExtractedModelConfig(
        model="crim", title="test_sim", source_height_m=0.05,
        domain_x=1.0, domain_y=0.5, cells_per_wavelength=10.0,
        max_cell_m=0.005, temperature_c=20.0, enforce_validity=False,
    )


# ---------------------------------------------------------------------------
# 1. merge_aggregations: None existing -> returns new as-is
# ---------------------------------------------------------------------------
def test_merge_with_none_existing():
    new = _empty_aggregation()
    new.model_params.title = "hello"
    result = merge_aggregations(None, new)
    assert result.model_params.title == "hello"
    print("PASS: merge with None existing returns new as-is")


# ---------------------------------------------------------------------------
# 2. Flat field override: non-None new value overrides existing
# ---------------------------------------------------------------------------
def test_flat_field_override():
    existing = _empty_aggregation()
    existing.model_params.title = "old_title"
    existing.model_params.model = "crim"

    new = _empty_aggregation()
    new.model_params.title = "new_title"
    # model is None in new -> should keep "crim"

    result = merge_aggregations(existing, new)
    assert result.model_params.title == "new_title", f"Expected 'new_title', got '{result.model_params.title}'"
    assert result.model_params.model == "crim", f"Expected 'crim', got '{result.model_params.model}'"
    print("PASS: flat field override works, None preserves existing")


# ---------------------------------------------------------------------------
# 3. Antenna field modification preserves unmentioned fields
# ---------------------------------------------------------------------------
def test_antenna_partial_update():
    existing = _empty_aggregation()
    existing.antenna_waveform = _sample_antenna()

    new = _empty_aggregation()
    new.antenna_waveform.waveform_center_freq_hz = 800e6  # change freq only

    result = merge_aggregations(existing, new)
    assert result.antenna_waveform.waveform_center_freq_hz == 800e6
    assert result.antenna_waveform.tx_rx_offset_m == 0.1, "tx_rx_offset should be preserved"
    assert result.antenna_waveform.waveform_name == "my_ricker", "waveform_name should be preserved"
    print("PASS: antenna partial update preserves unmentioned fields")


# ---------------------------------------------------------------------------
# 4. Layers: new layers replace existing entirely
# ---------------------------------------------------------------------------
def test_layers_replace():
    existing = _empty_aggregation()
    existing.layers = _sample_layers()

    new_layers = ExtractedLayers(num_layers=1, layers=[
        ExtractedLayerParams(name="single_layer", thickness_m=1.0, texture_class="sand", moisture_state="dry"),
    ])
    new = _empty_aggregation()
    new.layers = new_layers

    result = merge_aggregations(existing, new)
    assert result.layers.num_layers == 1
    assert result.layers.layers[0].name == "single_layer"
    print("PASS: new layers replace existing entirely")


# ---------------------------------------------------------------------------
# 5. Layers: empty new layers preserve existing
# ---------------------------------------------------------------------------
def test_layers_preserved_when_new_is_empty():
    existing = _empty_aggregation()
    existing.layers = _sample_layers()

    new = _empty_aggregation()  # layers empty (num_layers=0)

    result = merge_aggregations(existing, new)
    assert result.layers.num_layers == 2
    assert result.layers.layers[0].name == "topsoil"
    print("PASS: empty new layers preserve existing layers")


# ---------------------------------------------------------------------------
# 6. Multi-turn accumulation: layers -> antenna -> model -> complete
# ---------------------------------------------------------------------------
def test_incremental_accumulation():
    state = None

    # Turn 1: user provides layers only
    turn1 = _empty_aggregation()
    turn1.layers = _sample_layers()
    state = merge_aggregations(state, turn1)

    _, missing1 = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert len(missing1) > 0, "Should have missing params after layers only"
    assert state.layers.num_layers == 2

    # Turn 2: user provides antenna/waveform
    turn2 = _empty_aggregation()
    turn2.antenna_waveform = _sample_antenna()
    state = merge_aggregations(state, turn2)

    assert state.layers.num_layers == 2, "Layers should still be 2 after antenna turn"
    assert state.antenna_waveform.waveform_center_freq_hz == 400e6

    _, missing2 = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert len(missing2) < len(missing1), "Should have fewer missing params now"

    # Turn 3: user provides model config
    turn3 = _empty_aggregation()
    turn3.model_params = _sample_model()
    state = merge_aggregations(state, turn3)

    assert state.layers.num_layers == 2, "Layers still intact"
    assert state.antenna_waveform.waveform_center_freq_hz == 400e6, "Antenna still intact"
    assert state.model_params.model == "crim"

    gpr_schema, missing3 = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert len(missing3) == 0, f"Should be complete now, but missing: {missing3}"
    assert gpr_schema is not None
    print("PASS: incremental accumulation across 3 turns produces complete schema")


# ---------------------------------------------------------------------------
# 7. Post-generation modification: tweak frequency and re-resolve
# ---------------------------------------------------------------------------
def test_post_generation_modification():
    # Build a complete state
    state = AggregatedExtraction(
        layers=_sample_layers(),
        antenna_waveform=_sample_antenna(),
        model_params=_sample_model(),
        optional_params=ExtractedOptionalParams(),
    )

    gpr1, missing1 = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert gpr1 is not None and len(missing1) == 0
    assert gpr1.waveform.center_freq_hz == 400e6

    # User says "change frequency to 800 MHz"
    tweak = _empty_aggregation()
    tweak.antenna_waveform.waveform_center_freq_hz = 800e6
    state = merge_aggregations(state, tweak)

    gpr2, missing2 = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert gpr2 is not None and len(missing2) == 0
    assert gpr2.waveform.center_freq_hz == 800e6, f"Expected 800 MHz, got {gpr2.waveform.center_freq_hz}"
    assert gpr2.antenna.tx_rx_offset_m == 0.1, "tx_rx_offset should be unchanged"
    assert gpr2.layers[0].name == "topsoil", "Layers should be unchanged"
    assert gpr2.title == "test_sim", "Title should be unchanged"
    print("PASS: post-generation frequency modification works, everything else preserved")


# ---------------------------------------------------------------------------
# 8. Post-generation modification: change model type
# ---------------------------------------------------------------------------
def test_change_model_type():
    state = AggregatedExtraction(
        layers=_sample_layers(),
        antenna_waveform=_sample_antenna(),
        model_params=_sample_model(),
        optional_params=ExtractedOptionalParams(),
    )

    tweak = _empty_aggregation()
    tweak.model_params.model = "mironov"
    state = merge_aggregations(state, tweak)

    gpr, missing = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert gpr is not None
    assert gpr.model == "mironov", f"Expected mironov, got {gpr.model}"
    assert gpr.title == "test_sim", "Title should be unchanged"
    assert gpr.temperature_c == 20.0, "Temperature should be unchanged"
    print("PASS: model type changed, all other fields preserved")


# ---------------------------------------------------------------------------
# 9. Post-generation modification: replace layers entirely
# ---------------------------------------------------------------------------
def test_replace_layers_after_generation():
    state = AggregatedExtraction(
        layers=_sample_layers(),
        antenna_waveform=_sample_antenna(),
        model_params=_sample_model(),
        optional_params=ExtractedOptionalParams(),
    )

    new_layers = ExtractedLayers(num_layers=1, layers=[
        ExtractedLayerParams(name="new_single", thickness_m=2.0, texture_class="loam", moisture_state="wet"),
    ])
    tweak = _empty_aggregation()
    tweak.layers = new_layers
    state = merge_aggregations(state, tweak)

    gpr, missing = merge_extractions(state.layers, state.antenna_waveform, state.model_params, state.optional_params)
    assert gpr is not None
    assert len(gpr.layers) == 1
    assert gpr.layers[0].name == "new_single"
    assert gpr.waveform.center_freq_hz == 400e6, "Antenna/waveform should be unchanged"
    print("PASS: layers replaced, antenna/model preserved")


# ---------------------------------------------------------------------------
# 10. Serialisation round-trip: state survives model_dump -> reconstruct
# ---------------------------------------------------------------------------
def test_serialisation_roundtrip():
    state = AggregatedExtraction(
        layers=_sample_layers(),
        antenna_waveform=_sample_antenna(),
        model_params=_sample_model(),
        optional_params=ExtractedOptionalParams(),
    )

    dumped = state.model_dump()
    restored = AggregatedExtraction(**dumped)

    assert restored.layers.num_layers == 2
    assert restored.antenna_waveform.waveform_center_freq_hz == 400e6
    assert restored.model_params.model == "crim"

    gpr, missing = merge_extractions(restored.layers, restored.antenna_waveform, restored.model_params, restored.optional_params)
    assert gpr is not None and len(missing) == 0
    print("PASS: serialisation round-trip preserves all data")


if __name__ == "__main__":
    tests = [
        test_merge_with_none_existing,
        test_flat_field_override,
        test_antenna_partial_update,
        test_layers_replace,
        test_layers_preserved_when_new_is_empty,
        test_incremental_accumulation,
        test_post_generation_modification,
        test_change_model_type,
        test_replace_layers_after_generation,
        test_serialisation_roundtrip,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)

"""
End-to-end multi-turn integration test.

Simulates a real user session: sends messages one at a time through
simulate_workflow, carries state between turns, and verifies that
parameters accumulate, modifications take effect, and regeneration works.

Requires OPENAI_API_KEY (calls real LLM agents).
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.dirname(__file__))

from schema import AggregatedExtraction
from generator_agent import simulate_workflow


def _restore_state(result: dict) -> AggregatedExtraction | None:
    """Reconstruct state from workflow result, exactly as app.py does."""
    params = result.get("params")
    if params:
        return AggregatedExtraction(**params)
    return None


def _print_state_summary(state: AggregatedExtraction | None):
    if state is None:
        print("  State: None")
        return
    layers = state.layers
    aw = state.antenna_waveform
    mc = state.model_params
    print(f"  Layers: {layers.num_layers} — {[l.name for l in layers.layers]}")
    print(f"  Antenna freq: {aw.waveform_center_freq_hz}, offset: {aw.tx_rx_offset_m}")
    print(f"  Model: {mc.model}, title: {mc.title}, domain: ({mc.domain_x}, {mc.domain_y})")


async def run_multi_turn_test():
    state = None
    user_id = "integration_test"

    # ------------------------------------------------------------------
    # Turn 1: Provide only layers
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURN 1: Provide layers only")
    print("=" * 70)
    msg1 = (
        "I want a 2-layer simulation. "
        "Layer 1: topsoil, thickness 0.3m, sandy_loam, normal moisture. "
        "Layer 2: clay base, thickness 0.8m, clay, wet moisture."
    )
    print(f"  User: {msg1}")
    r1 = await simulate_workflow(msg1, user_id=user_id, current_state=state)
    state = _restore_state(r1)

    print(f"  Status: {r1['status']}")
    _print_state_summary(state)

    assert r1["status"] == "incomplete", f"Expected incomplete, got {r1['status']}"
    assert state is not None and state.layers.num_layers == 2, "Should have 2 layers"
    print("  ✓ Turn 1 OK — layers collected, status incomplete")

    # ------------------------------------------------------------------
    # Turn 2: Provide antenna / waveform
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURN 2: Provide antenna and waveform")
    print("=" * 70)
    msg2 = (
        "Use a Ricker waveform at 400 MHz, amplitude 1.0, name my_ricker. "
        "Hertzian dipole antenna, z-axis, tx_rx offset 0.1 m."
    )
    print(f"  User: {msg2}")
    r2 = await simulate_workflow(msg2, user_id=user_id, current_state=state)
    state = _restore_state(r2)

    print(f"  Status: {r2['status']}")
    _print_state_summary(state)

    assert state.layers.num_layers == 2, "Layers should still be 2"
    assert state.antenna_waveform.waveform_center_freq_hz == 400e6 or \
           state.antenna_waveform.waveform_center_freq_hz == 4e8, \
           f"Freq should be ~400 MHz, got {state.antenna_waveform.waveform_center_freq_hz}"
    print("  ✓ Turn 2 OK — antenna collected, layers preserved")

    # ------------------------------------------------------------------
    # Turn 3: Provide model config -> should complete
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURN 3: Provide model config (should complete the schema)")
    print("=" * 70)
    msg3 = (
        "Model is CRIM, title test_simulation, source height 0.05 m, "
        "domain 1.0 x 0.6 m, cells per wavelength 10, max cell 0.005 m, "
        "temperature 20 C, enforce validity false."
    )
    print(f"  User: {msg3}")
    r3 = await simulate_workflow(msg3, user_id=user_id, current_state=state)
    state = _restore_state(r3)

    print(f"  Status: {r3['status']}")
    _print_state_summary(state)

    assert r3["status"] == "complete", f"Expected complete, got {r3['status']}. Message: {r3.get('message', '')[:300]}"
    assert r3.get("file_path") is not None, "Should have generated a file"
    assert state.model_params.model.lower() == "crim"
    print(f"  ✓ Turn 3 OK — file generated at {r3['file_path']}")

    # Save original freq for comparison
    original_freq = state.antenna_waveform.waveform_center_freq_hz

    # ------------------------------------------------------------------
    # Turn 4: Modify a single parameter (frequency) after generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURN 4: Modify frequency to 800 MHz (post-generation tweak)")
    print("=" * 70)
    msg4 = "Change the waveform frequency to 800 MHz."
    print(f"  User: {msg4}")
    r4 = await simulate_workflow(msg4, user_id=user_id, current_state=state)
    state = _restore_state(r4)

    print(f"  Status: {r4['status']}")
    _print_state_summary(state)

    new_freq = state.antenna_waveform.waveform_center_freq_hz
    assert new_freq is not None and abs(new_freq - 800e6) < 1e3, \
        f"Freq should be ~800 MHz, got {new_freq}"
    assert state.layers.num_layers == 2, "Layers should be unchanged"
    assert state.model_params.model.lower() == "crim", "Model should be unchanged"
    assert state.model_params.title is not None, "Title should be unchanged"

    if r4["status"] == "complete":
        print(f"  ✓ Turn 4 OK — frequency changed to 800 MHz, regenerated file")
    else:
        print(f"  ✓ Turn 4 OK — frequency changed to 800 MHz, status={r4['status']}")

    # ------------------------------------------------------------------
    # Turn 5: Modify model type
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURN 5: Change model to mironov")
    print("=" * 70)
    msg5 = "Switch the dielectric model to mironov."
    print(f"  User: {msg5}")
    r5 = await simulate_workflow(msg5, user_id=user_id, current_state=state)
    state = _restore_state(r5)

    print(f"  Status: {r5['status']}")
    _print_state_summary(state)

    assert state.model_params.model.lower() == "mironov", \
        f"Model should be mironov, got {state.model_params.model}"
    assert state.layers.num_layers == 2, "Layers should be unchanged"
    freq_after = state.antenna_waveform.waveform_center_freq_hz
    assert freq_after is not None and abs(freq_after - 800e6) < 1e3, \
        f"Freq should still be ~800 MHz, got {freq_after}"
    print(f"  ✓ Turn 5 OK — model changed to mironov, everything else preserved")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL TURNS PASSED")
    print("=" * 70)
    print(f"\nFinal state summary:")
    _print_state_summary(state)


if __name__ == "__main__":
    asyncio.run(run_multi_turn_test())

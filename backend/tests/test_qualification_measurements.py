"""Scientific comparison controls must hold independently of solver success."""
import pytest

from backend.qualify_3d import make_deck, refinement_controls


def _executed_like_record(dx):
    deck = make_deck("fixture", dx=dx, pml=round(.024 / dx))
    lines = {line.split(":", 1)[0]: line.split(":", 1)[1].split()
             for line in deck.splitlines()}
    return {
        "spacing": (dx, dx, dx),
        "source_m": list(map(float, lines["#hertzian_dipole"][1:4])),
        "receiver_m": list(map(float, lines["#rx"][:3])),
    }, float(lines["#waveform"][1])


def test_odd_refinement_preserves_physical_locations_moment_and_absorber_width():
    coarse, amplitude = _executed_like_record(.004)
    fine, fine_amplitude = _executed_like_record(.004 / 3)
    controls = refinement_controls(coarse, fine, 6, 18)
    assert controls["source_Ez_point_m"] == pytest.approx([.14, .14, .142])
    assert controls["receiver_Ez_point_m"] == pytest.approx([.172, .164, .158])
    assert controls["pml_thickness_xyz_m"] == pytest.approx([.024] * 3)
    assert amplitude * .004 == pytest.approx(fine_amplitude * .004 / 3)
    # Both decks must also have integer source and receiver anchors.
    for entry in (coarse, fine):
        for key in ("source_m", "receiver_m"):
            for value in entry[key]:
                cells = value / entry["spacing"][0]
                assert cells == pytest.approx(round(cells))


@pytest.mark.parametrize("kind", ["source", "receiver"])
def test_changed_native_measurement_location_fails_refinement(kind):
    coarse, _ = _executed_like_record(.004)
    fine, _ = _executed_like_record(.004 / 3)
    fine[kind + "_m"][2] -= .004 / 3
    with pytest.raises(ValueError, match="physical .* Ez location"):
        refinement_controls(coarse, fine, 6, 18)


def test_same_pml_cell_count_is_not_same_physical_absorber():
    coarse, _ = _executed_like_record(.004)
    fine, _ = _executed_like_record(.004 / 3)
    with pytest.raises(ValueError, match="physical PML thickness"):
        refinement_controls(coarse, fine, 6, 6)

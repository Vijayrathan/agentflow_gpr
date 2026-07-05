import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api
from backend.schema import DatasetConfig, ExtractedAntenna, ExtractedWaveform


def _sampled_manifest():
    return {
        "samples": [
            {
                "sample_id": 1,
                "layers": [
                    {
                        "name": "sand",
                        "thickness_m": 0.4,
                        "sand_pct": 80.0,
                        "clay_pct": 5.0,
                        "silt_pct": 15.0,
                        "theta_v_min": 0.05,
                        "theta_v_max": 0.15,
                        "bulk_density_gcm3": 1.5,
                        "particle_density_gcm3": 2.66,
                    }
                ],
                "targets": [
                    {
                        "kind": "cylinder",
                        "name": "target",
                        "material": "pec",
                        "x_offset_m": -0.1,
                        "depth_m": 0.2,
                        "radius_m": 0.05,
                    },
                    {
                        "kind": "box",
                        "name": "slab",
                        "material": "pec",
                        "x_offset_m": 0.2,
                        "depth_m": 0.3,
                        "width_m": 0.2,
                        "height_m": 0.06,
                    },
                ],
            }
        ]
    }


def _global_derive():
    return {
        "source_height_m": 0.25,
        "domain_x_m": 1.2,
        "domain_y_m": 0.9,
        "dx_m": 0.002,
    }


def test_build_simulation_rows_maps_emitted_file_and_sample_target():
    cfg = DatasetConfig(num_samples=1, model_basename="demo", output_dir="./dataset/demo")
    wf = ExtractedWaveform(
        waveform_kind="ricker",
        waveform_amplitude=1.0,
        waveform_center_freq_hz=400e6,
        waveform_name="ricker_400mhz",
    )
    ant = ExtractedAntenna(tx_rx_offset_m=0.12, antenna_axis="x")
    session_id = uuid.uuid4()

    rows = api._build_simulation_rows(
        session_uuid=session_id,
        user_id="user-1",
        cfg=cfg,
        wf=wf,
        ant=ant,
        adv=None,
        sampled_manifest=_sampled_manifest(),
        global_derive=_global_derive(),
        emitted_manifest={"files": [{"sample_id": 1, "path": "/tmp/demo_1.in"}]},
    )

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row["id"], uuid.UUID)
    assert row["session_id"] == session_id
    assert row["sample_index"] == 1
    assert row["model"] == "demo"
    assert row["input_file_path"] == "/tmp/demo_1.in"
    assert row["max_cell_m"] == 0.002
    # per-kind split of the sample's targets; no adv geometry exists any more
    assert row["cylinders"] == [_sampled_manifest()["samples"][0]["targets"][0]]
    assert row["boxes"] == [_sampled_manifest()["samples"][0]["targets"][1]]
    assert row["spheres"] is None
    assert row["layers"][0]["name"] == "sand"


def test_finalize_dataset_sync_upserts_session_and_batches_rows(tmp_path, monkeypatch):
    sampled = tmp_path / "sampled_layers.json"
    global_path = tmp_path / "global_derive.json"
    emitted = tmp_path / "emitted_files.json"
    sampled.write_text(api.json.dumps(_sampled_manifest()), encoding="utf-8")
    global_path.write_text(api.json.dumps(_global_derive()), encoding="utf-8")
    emitted.write_text(
        api.json.dumps({"in_dir": str(tmp_path / "in_files"), "n_written": 1, "files": [{"sample_id": 1, "path": str(tmp_path / "demo_1.in")}], "errors": []}),
        encoding="utf-8",
    )

    class FakeDb:
        def __init__(self):
            self.row = None
            self.commits = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, _model, _key):
            return self.row

        def add(self, row):
            self.row = row

        def commit(self):
            self.commits += 1

    fake_db = FakeDb()
    inserted_rows = []
    monkeypatch.setattr(api, "get_session", lambda: fake_db)
    monkeypatch.setattr(api, "batch_insert_simulations", lambda rows: inserted_rows.extend(rows) or len(rows))

    payload = api.FinalizeDatasetPayload(
        session_id=str(uuid.uuid4()),
        user_id="user-1",
        dataset_config=DatasetConfig(num_samples=1, model_basename="demo", output_dir=str(tmp_path)).model_dump(),
        layers={"num_layers": 1, "layers": []},
        target_ranges=None,
        waveform=ExtractedWaveform(
            waveform_kind="ricker",
            waveform_amplitude=1.0,
            waveform_center_freq_hz=400e6,
            waveform_name="ricker_400mhz",
        ).model_dump(),
        antenna=ExtractedAntenna(tx_rx_offset_m=0.12, antenna_axis="x").model_dump(),
        advanced_params=None,
        artifacts={
            "output_dir": str(tmp_path),
            "in_dir": str(tmp_path / "in_files"),
            "sampled_layers_json": str(sampled),
            "global_derive_json": str(global_path),
            "emitted_files_json": str(emitted),
        },
        emission={"num_requested": 1, "num_generated": 1, "num_failed": 0, "errors": []},
    )

    result = api._finalize_dataset_sync(payload)

    assert result["status"] == "complete"
    assert result["rows_inserted"] == 1
    assert fake_db.row.status == "complete"
    assert fake_db.row.num_samples_requested == 1
    assert fake_db.row.model_config_data["dataset_config"]["model_basename"] == "demo"
    assert inserted_rows[0]["input_file_path"].endswith("demo_1.in")

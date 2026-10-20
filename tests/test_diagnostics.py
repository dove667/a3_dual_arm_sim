import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic


def test_diagnostic_video_uses_h264(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required for codec verification")
    diagnostic = EpisodeDiagnostic(tmp_path / "trace")
    for value in (0, 127, 255):
        diagnostic.add_frame(
            {"observation.images.front": np.full((32, 32, 3), value, dtype=np.uint8)},
            np.zeros(16),
            {"applied_action": np.zeros(16)},
        )
    diagnostic.close()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,nb_frames",
            "-of",
            "json",
            str(tmp_path / "trace/front.mp4"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["nb_frames"] == "3"


def test_failed_conversion_preserves_original_video(tmp_path, monkeypatch):
    from a3_dual_arm_sim.workflows.diagnostics import _encode_preview_video

    path = tmp_path / "front.mp4"
    path.write_bytes(b"original video")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "encoder failed"),
    )
    with pytest.raises(RuntimeError, match="encoder failed"):
        _encode_preview_video(path, "ffmpeg")
    assert path.read_bytes() == b"original video"
    assert list(tmp_path.iterdir()) == [path]


def test_trace_keeps_before_observation_and_distinct_actions(tmp_path):
    diagnostic = EpisodeDiagnostic(tmp_path / "trace")
    observation = {
        "observation.state": np.arange(16),
        "observation.images.front": np.full((32, 32, 3), 127, dtype=np.uint8),
    }
    diagnostic.add_frame(
        observation,
        np.ones(16),
        {"applied_action": np.full(16, 2), "cookies_in_target": 5},
        raw_action=np.zeros(16),
        phase="RELEASE",
    )
    diagnostic.close()
    row = json.loads((tmp_path / "trace/trace.jsonl").read_text())
    assert row["observation_before"]["observation.state"] == list(range(16))
    assert row["raw_action"] == [0] * 16
    assert row["processed_action"] == [1] * 16
    assert row["applied_action"] == [2] * 16
    reader = cv2.VideoCapture(str(tmp_path / "trace/front.mp4"))
    ok, frame = reader.read()
    reader.release()
    assert ok and frame.shape == (32, 32, 3)


def test_trace_identifies_missing_slots_and_boundary(tmp_path):
    diagnostic = EpisodeDiagnostic(tmp_path / "trace")
    diagnostic.add_frame(
        {"observation.state": np.zeros(16)},
        np.zeros(16),
        {
            "applied_action": np.zeros(16),
            "cookies_in_target": 10,
            "cookies_in_source": 70,
            "count_fill": True,
            "target_slot_occupancy": [0, 1, 2, 3, 4, 5, 6, 7, 8, -1],
            "target_touches_all_walls": False,
            "exact_2x5_fill": False,
            "success_criterion": "exact_slots",
            "success_hold_count": 0,
            "required_success_hold_steps": 5,
            "success": False,
        },
    )
    diagnostic.close()
    row = json.loads((tmp_path / "trace/trace.jsonl").read_text())
    assert row["after"]["unmet_success_conditions"] == [
        "slot_occupancy",
        "boundary_coverage",
        "hold_steps",
    ]
    summary = json.loads((tmp_path / "trace/success_diagnostics.json").read_text())
    assert summary["first_count_fill_step"] == 0
    assert summary["first_exact_fill_step"] is None
    assert summary["max_success_hold_count"] == 0

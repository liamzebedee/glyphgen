"""
Tests for persistence layer.

Tests cover:
- Font ID validation
- Directory creation
- Save/load round-trip numerical equivalence
- Run discovery and listing
- Corruption detection
- Concurrent access safety
- Atomic save operations
"""

import json
import pytest
import shutil
import tempfile
import torch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.config import Config
from src.glyphnet import GlyphNetwork
from src.persistence import (
    FONT_ID_PATTERN,
    WEIGHTS_VERSION,
    RunMetadata,
    delete_font,
    delete_run,
    ensure_font_dirs,
    export_style_z,
    generate_run_id,
    get_font_dir,
    get_latest_run,
    get_run_files,
    get_run_info,
    get_runs_dir,
    is_run_complete,
    list_fonts,
    list_runs,
    load_eval_glyphs,
    load_feedback,
    load_model,
    load_sample_image,
    save_model,
    validate_font_id,
)


@pytest.fixture
def temp_networks_dir(tmp_path):
    """Create a temporary networks directory for testing."""
    networks_dir = tmp_path / "networks"
    networks_dir.mkdir()

    # Patch CONFIG to use temp directory
    config = Config(project_root=tmp_path)

    with patch("src.persistence.CONFIG", config):
        yield networks_dir


@pytest.fixture
def model():
    """Create a GlyphNetwork model for testing."""
    return GlyphNetwork()


@pytest.fixture
def style_z():
    """Create a style vector for testing."""
    return torch.randn(64)


class TestFontIdValidation:
    """Font ID validation tests."""

    def test_valid_font_ids(self):
        """Test valid font IDs are accepted."""
        valid_ids = [
            "1-fruity",
            "2-dumb",
            "3-aggressive-sans",
            "10-elegant",
            "123-test-font-name",
            "0-a",
        ]
        for font_id in valid_ids:
            assert validate_font_id(font_id), f"'{font_id}' should be valid"

    def test_invalid_font_ids(self):
        """Test invalid font IDs are rejected."""
        invalid_ids = [
            "fruity",  # Missing number
            "1fruity",  # Missing hyphen
            "1-",  # Missing name
            "-fruity",  # Missing number
            "1-Fruity",  # Uppercase
            "1-fruity_bold",  # Underscore
            "1-fruity bold",  # Space
            "one-fruity",  # Non-numeric prefix
            "",  # Empty
            "1-1bad",  # Name starts with number
        ]
        for font_id in invalid_ids:
            assert not validate_font_id(font_id), f"'{font_id}' should be invalid"

    def test_font_id_pattern_regex(self):
        """Test the regex pattern directly."""
        assert FONT_ID_PATTERN.match("1-fruity")
        assert FONT_ID_PATTERN.match("99-a")
        assert not FONT_ID_PATTERN.match("1-")
        assert not FONT_ID_PATTERN.match("abc")


class TestRunIdGeneration:
    """Run ID generation tests."""

    def test_generate_run_id_format(self):
        """Test run ID is ISO 8601 format with microseconds."""
        run_id = generate_run_id()

        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(run_id)
        assert dt is not None

        # Should include microseconds
        assert "." in run_id
        assert len(run_id.split(".")[-1]) == 6

    def test_generate_run_id_uniqueness(self):
        """Test sequential run IDs are unique."""
        run_ids = [generate_run_id() for _ in range(100)]
        assert len(set(run_ids)) == len(run_ids)

    def test_generate_run_id_chronological(self):
        """Test run IDs are chronologically sortable."""
        run_ids = [generate_run_id() for _ in range(10)]

        # Sorted should be same as original order
        sorted_ids = sorted(run_ids)
        assert run_ids == sorted_ids


class TestDirectoryManagement:
    """Directory creation and management tests."""

    def test_ensure_font_dirs_creates_structure(self, temp_networks_dir):
        """Test directory structure is created."""
        runs_dir = ensure_font_dirs("1-fruity")

        assert runs_dir.exists()
        assert runs_dir.parent.name == "1-fruity"
        assert runs_dir.name == "runs"

    def test_get_font_dir_validates_id(self, temp_networks_dir):
        """Test get_font_dir validates font ID."""
        with pytest.raises(ValueError, match="Invalid font ID"):
            get_font_dir("invalid")

    def test_get_runs_dir(self, temp_networks_dir):
        """Test get_runs_dir returns correct path."""
        runs_dir = get_runs_dir("1-fruity")
        assert runs_dir == temp_networks_dir / "1-fruity" / "runs"


class TestListFonts:
    """Font listing tests."""

    def test_list_fonts_empty(self, temp_networks_dir):
        """Test listing empty networks directory."""
        fonts = list_fonts()
        assert fonts == []

    def test_list_fonts_single(self, temp_networks_dir):
        """Test listing single font."""
        ensure_font_dirs("1-fruity")
        fonts = list_fonts()
        assert fonts == ["1-fruity"]

    def test_list_fonts_multiple(self, temp_networks_dir):
        """Test listing multiple fonts."""
        ensure_font_dirs("1-fruity")
        ensure_font_dirs("2-dumb")
        ensure_font_dirs("3-aggressive")

        fonts = list_fonts()
        assert fonts == ["1-fruity", "2-dumb", "3-aggressive"]

    def test_list_fonts_ignores_invalid(self, temp_networks_dir):
        """Test listing ignores invalid font directories."""
        ensure_font_dirs("1-fruity")
        (temp_networks_dir / "invalid").mkdir()  # Not a valid font ID
        (temp_networks_dir / "some_file.txt").touch()  # Not a directory

        fonts = list_fonts()
        assert fonts == ["1-fruity"]


class TestSaveLoad:
    """Save and load round-trip tests."""

    def test_save_creates_files(self, temp_networks_dir, model, style_z):
        """Test save creates all expected files."""
        sample_img = Image.new("L", (128, 128), color=128)
        eval_glyphs = {"a": Image.new("L", (128, 128))}
        feedback = {"rating": 7, "comments": "good"}

        run_id = save_model(
            font_id="1-fruity",
            model=model,
            style_z=style_z,
            sample_image=sample_img,
            eval_glyphs=eval_glyphs,
            feedback=feedback,
        )

        run_dir = temp_networks_dir / "1-fruity" / "runs" / run_id
        assert run_dir.exists()
        assert (run_dir / "weights").exists()
        assert (run_dir / "sample.png").exists()
        assert (run_dir / "eval-glyphs" / "a.png").exists()
        assert (run_dir / "feedback.json").exists()
        assert (run_dir / "metadata.json").exists()

    def test_save_load_weights_equivalence(self, temp_networks_dir, model, style_z):
        """Test weights are numerically equivalent after load."""
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        original_style_z = style_z.clone()

        run_id = save_model(
            font_id="1-fruity",
            model=model,
            style_z=style_z,
        )

        # Reset model weights to verify load works
        for param in model.parameters():
            param.data.fill_(0)

        loaded_style_z, metadata = load_model("1-fruity", model, run_id)

        # Check weights are equivalent to 6 decimal places
        for key in original_state:
            loaded = model.state_dict()[key]
            original = original_state[key]
            assert torch.allclose(loaded, original, atol=1e-6), f"Mismatch in {key}"

        # Check style_z
        assert torch.allclose(loaded_style_z, original_style_z, atol=1e-6)

    def test_save_with_custom_run_id(self, temp_networks_dir, model, style_z):
        """Test save with custom run ID."""
        custom_id = "2026-01-01T00:00:00.000000"
        run_id = save_model(
            font_id="1-fruity",
            model=model,
            style_z=style_z,
            run_id=custom_id,
        )

        assert run_id == custom_id
        assert (temp_networks_dir / "1-fruity" / "runs" / custom_id).exists()

    def test_load_latest_run(self, temp_networks_dir, model, style_z):
        """Test load defaults to latest run."""
        # Create multiple runs
        save_model("1-fruity", model, style_z, run_id="2026-01-01T00:00:00.000000")
        save_model("1-fruity", model, style_z, run_id="2026-01-02T00:00:00.000000")
        save_model("1-fruity", model, style_z, run_id="2026-01-03T00:00:00.000000")

        # Load without specifying run_id
        _, metadata = load_model("1-fruity", model)

        assert metadata.run_id == "2026-01-03T00:00:00.000000"

    def test_load_nonexistent_font_raises(self, temp_networks_dir, model):
        """Test loading nonexistent font raises error."""
        with pytest.raises(FileNotFoundError, match="No runs found"):
            load_model("99-nonexistent", model)

    def test_load_nonexistent_run_raises(self, temp_networks_dir, model, style_z):
        """Test loading nonexistent run raises error."""
        save_model("1-fruity", model, style_z)

        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            load_model("1-fruity", model, run_id="nonexistent")


class TestRunListing:
    """Run listing and metadata tests."""

    def test_list_runs_empty(self, temp_networks_dir):
        """Test listing runs for font with no runs."""
        ensure_font_dirs("1-fruity")
        runs = list_runs("1-fruity")
        assert runs == []

    def test_list_runs_chronological(self, temp_networks_dir, model, style_z):
        """Test runs are listed chronologically."""
        ids = [
            "2026-01-03T00:00:00.000000",
            "2026-01-01T00:00:00.000000",
            "2026-01-02T00:00:00.000000",
        ]

        for run_id in ids:
            save_model("1-fruity", model, style_z, run_id=run_id)

        runs = list_runs("1-fruity")
        run_ids = [r.run_id for r in runs]

        # Should be oldest first
        assert run_ids == [
            "2026-01-01T00:00:00.000000",
            "2026-01-02T00:00:00.000000",
            "2026-01-03T00:00:00.000000",
        ]

    def test_get_run_info(self, temp_networks_dir, model, style_z):
        """Test get_run_info returns correct metadata."""
        training_params = {"lr": 0.001, "epochs": 100}
        run_id = save_model(
            "1-fruity",
            model,
            style_z,
            training_params=training_params,
        )

        info = get_run_info("1-fruity", run_id)

        assert info.run_id == run_id
        assert info.font_id == "1-fruity"
        assert info.completed is True
        assert info.training_params == training_params

    def test_get_latest_run(self, temp_networks_dir, model, style_z):
        """Test get_latest_run returns most recent."""
        save_model("1-fruity", model, style_z, run_id="2026-01-01T00:00:00.000000")
        save_model("1-fruity", model, style_z, run_id="2026-01-02T00:00:00.000000")

        latest = get_latest_run("1-fruity")

        assert latest.run_id == "2026-01-02T00:00:00.000000"

    def test_get_latest_run_no_runs(self, temp_networks_dir):
        """Test get_latest_run returns None for empty font."""
        ensure_font_dirs("1-fruity")
        assert get_latest_run("1-fruity") is None


class TestCorruptionDetection:
    """Corruption and error handling tests."""

    def test_corrupted_weights_detected(self, temp_networks_dir, model, style_z):
        """Test corrupted weights file raises RuntimeError."""
        run_id = save_model("1-fruity", model, style_z)

        # Corrupt the weights file
        weights_path = temp_networks_dir / "1-fruity" / "runs" / run_id / "weights"
        with open(weights_path, "wb") as f:
            f.write(b"corrupted data")

        with pytest.raises(RuntimeError, match="Corrupted weights file"):
            load_model("1-fruity", model, run_id)

    def test_missing_keys_detected(self, temp_networks_dir, model, style_z):
        """Test weights missing required keys raises error."""
        run_id = save_model("1-fruity", model, style_z)

        # Overwrite with incomplete data
        weights_path = temp_networks_dir / "1-fruity" / "runs" / run_id / "weights"
        torch.save({"something_else": "data"}, weights_path)

        with pytest.raises(RuntimeError, match="missing keys"):
            load_model("1-fruity", model, run_id)

    def test_corrupted_metadata_handled(self, temp_networks_dir, model, style_z):
        """Test corrupted metadata is handled gracefully."""
        run_id = save_model("1-fruity", model, style_z)

        # Corrupt metadata
        metadata_path = (
            temp_networks_dir / "1-fruity" / "runs" / run_id / "metadata.json"
        )
        with open(metadata_path, "w") as f:
            f.write("not valid json")

        # list_runs should still work
        runs = list_runs("1-fruity")
        assert len(runs) == 1


class TestRunFiles:
    """Run file checking tests."""

    def test_get_run_files(self, temp_networks_dir, model, style_z):
        """Test get_run_files returns correct status."""
        run_id = save_model(
            "1-fruity",
            model,
            style_z,
            sample_image=Image.new("L", (128, 128)),
            eval_glyphs={"a": Image.new("L", (128, 128))},
            feedback={"rating": 5},
        )

        files = get_run_files("1-fruity", run_id)

        assert files["weights"] is True
        assert files["sample.png"] is True
        assert files["eval-glyphs"] is True
        assert files["feedback.json"] is True
        assert files["metadata.json"] is True

    def test_is_run_complete(self, temp_networks_dir, model, style_z):
        """Test is_run_complete checks weights existence."""
        run_id = save_model("1-fruity", model, style_z)

        assert is_run_complete("1-fruity", run_id) is True

        # Create incomplete run (no weights)
        incomplete_dir = temp_networks_dir / "1-fruity" / "runs" / "incomplete"
        incomplete_dir.mkdir()

        assert is_run_complete("1-fruity", "incomplete") is False


class TestDeletion:
    """Deletion tests."""

    def test_delete_run(self, temp_networks_dir, model, style_z):
        """Test deleting a run."""
        run_id = save_model("1-fruity", model, style_z)

        run_dir = temp_networks_dir / "1-fruity" / "runs" / run_id
        assert run_dir.exists()

        delete_run("1-fruity", run_id)

        assert not run_dir.exists()

    def test_delete_run_nonexistent(self, temp_networks_dir):
        """Test deleting nonexistent run raises error."""
        ensure_font_dirs("1-fruity")

        with pytest.raises(FileNotFoundError, match="not found"):
            delete_run("1-fruity", "nonexistent")

    def test_delete_font(self, temp_networks_dir, model, style_z):
        """Test deleting entire font."""
        save_model("1-fruity", model, style_z)

        font_dir = temp_networks_dir / "1-fruity"
        assert font_dir.exists()

        delete_font("1-fruity")

        assert not font_dir.exists()


class TestLoaders:
    """Additional loader tests."""

    def test_load_feedback(self, temp_networks_dir, model, style_z):
        """Test loading feedback data."""
        feedback = {"rating": 8, "notes": "excellent"}
        run_id = save_model("1-fruity", model, style_z, feedback=feedback)

        loaded = load_feedback("1-fruity", run_id)

        assert loaded == feedback

    def test_load_feedback_missing(self, temp_networks_dir, model, style_z):
        """Test loading missing feedback returns None."""
        run_id = save_model("1-fruity", model, style_z)

        assert load_feedback("1-fruity", run_id) is None

    def test_load_sample_image(self, temp_networks_dir, model, style_z):
        """Test loading sample image."""
        img = Image.new("L", (128, 128), color=100)
        run_id = save_model("1-fruity", model, style_z, sample_image=img)

        loaded = load_sample_image("1-fruity", run_id)

        assert loaded is not None
        assert loaded.size == (128, 128)

    def test_load_eval_glyphs(self, temp_networks_dir, model, style_z):
        """Test loading evaluation glyphs."""
        glyphs = {
            "a": Image.new("L", (128, 128)),
            "b": Image.new("L", (128, 128)),
            "c": Image.new("L", (128, 128)),
        }
        run_id = save_model("1-fruity", model, style_z, eval_glyphs=glyphs)

        loaded = load_eval_glyphs("1-fruity", run_id)

        assert set(loaded.keys()) == {"a", "b", "c"}


class TestExportStyleZ:
    """Style vector export tests."""

    def test_export_style_z(self, temp_networks_dir, model, style_z):
        """Test exporting style_z to file."""
        run_id = save_model("1-fruity", model, style_z)

        export_path = temp_networks_dir / "exported_style.pt"
        export_style_z("1-fruity", run_id, export_path)

        assert export_path.exists()

        loaded = torch.load(export_path, weights_only=False)
        assert "style_z" in loaded
        assert torch.allclose(loaded["style_z"], style_z, atol=1e-6)


class TestRunMetadata:
    """RunMetadata dataclass tests."""

    def test_to_dict(self):
        """Test RunMetadata serialization."""
        metadata = RunMetadata(
            run_id="2026-01-01T00:00:00.000000",
            font_id="1-fruity",
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1, 1, 0),
            completed=True,
            training_params={"lr": 0.001},
        )

        d = metadata.to_dict()

        assert d["run_id"] == "2026-01-01T00:00:00.000000"
        assert d["font_id"] == "1-fruity"
        assert d["completed"] is True

    def test_from_dict(self):
        """Test RunMetadata deserialization."""
        d = {
            "run_id": "2026-01-01T00:00:00.000000",
            "font_id": "1-fruity",
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T01:00:00",
            "completed": True,
            "training_params": {"lr": 0.001},
        }

        metadata = RunMetadata.from_dict(d)

        assert metadata.run_id == "2026-01-01T00:00:00.000000"
        assert metadata.font_id == "1-fruity"
        assert metadata.completed is True

    def test_run_dir_property(self, temp_networks_dir):
        """Test run_dir property returns correct path."""
        metadata = RunMetadata(
            run_id="2026-01-01T00:00:00.000000",
            font_id="1-fruity",
            start_time=datetime(2026, 1, 1),
        )

        expected = temp_networks_dir / "1-fruity" / "runs" / "2026-01-01T00:00:00.000000"
        assert metadata.run_dir == expected


class TestConcurrentAccess:
    """Concurrent access safety tests."""

    def test_concurrent_save_different_fonts(self, temp_networks_dir, model, style_z):
        """Test concurrent saves to different fonts don't conflict."""

        def save_to_font(font_id):
            return save_model(font_id, model, style_z.clone())

        font_ids = ["1-fruity", "2-dumb", "3-elegant", "4-bold"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(save_to_font, font_ids))

        assert len(results) == 4
        assert all(r is not None for r in results)

        # All fonts should exist
        fonts = list_fonts()
        assert set(fonts) == set(font_ids)


class TestWeightsVersion:
    """Weights versioning tests."""

    def test_weights_include_version(self, temp_networks_dir, model, style_z):
        """Test saved weights include version metadata."""
        run_id = save_model("1-fruity", model, style_z)

        weights_path = temp_networks_dir / "1-fruity" / "runs" / run_id / "weights"
        data = torch.load(weights_path, weights_only=False)

        assert data["weights_version"] == WEIGHTS_VERSION
        assert "timestamp" in data

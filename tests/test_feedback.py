"""
Tests for Claude Vision API feedback collection.

Tests cover image encoding integration, feedback storage, response parsing,
and error handling with mocked Claude API.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import CONFIG


# Test fixtures
@pytest.fixture
def sample_glyph():
    """Create a sample glyph tensor."""
    return torch.rand(1, 128, 128)


@pytest.fixture
def sample_glyphs():
    """Create a dict of sample glyphs for all lowercase letters."""
    return {chr(ord("a") + i): torch.rand(1, 128, 128) for i in range(26)}


@pytest.fixture
def temp_networks_dir(tmp_path):
    """Create a temporary networks directory."""
    networks_dir = tmp_path / "networks"
    networks_dir.mkdir()

    # Temporarily override CONFIG.networks_dir
    original_networks_dir = CONFIG.networks_dir
    # We need to patch the property through the class
    with patch.object(type(CONFIG), "networks_dir", property(lambda self: networks_dir)):
        yield networks_dir

    # Restore original (though context manager handles cleanup)


@pytest.fixture
def mock_anthropic():
    """Mock the anthropic client."""
    # Create a mock anthropic module
    mock_module = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"fruity": 7, "aggressive": 3, "dumb": 5, "elegant": 6}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_module.Anthropic.return_value = mock_client

    # Patch the import statement in the module
    import sys
    sys.modules["anthropic"] = mock_module

    yield mock_module

    # Cleanup - restore original if it existed
    if "anthropic" in sys.modules and sys.modules["anthropic"] is mock_module:
        del sys.modules["anthropic"]


class TestFeedbackResult:
    """Tests for FeedbackResult dataclass."""

    def test_create_feedback_result(self):
        """Test creating a feedback result."""
        from src.feedback import FeedbackResult

        result = FeedbackResult(
            glyph_char="a",
            ratings={"fruity": 7.0, "aggressive": 3.0},
            timestamp=datetime.now(),
            model="claude-sonnet-4-20250514",
            success=True,
        )
        assert result.glyph_char == "a"
        assert result.ratings["fruity"] == 7.0
        assert result.success is True

    def test_feedback_result_to_dict(self):
        """Test converting feedback result to dictionary."""
        from src.feedback import FeedbackResult

        timestamp = datetime.now()
        result = FeedbackResult(
            glyph_char="b",
            ratings={"fruity": 5.0},
            timestamp=timestamp,
            model="test-model",
            success=True,
        )
        d = result.to_dict()
        assert d["glyph_char"] == "b"
        assert d["ratings"] == {"fruity": 5.0}
        assert d["timestamp"] == timestamp.isoformat()
        assert d["success"] is True

    def test_feedback_result_from_dict(self):
        """Test creating feedback result from dictionary."""
        from src.feedback import FeedbackResult

        data = {
            "glyph_char": "c",
            "ratings": {"aggressive": 8.0},
            "timestamp": "2026-01-11T10:00:00",
            "model": "test-model",
            "success": True,
            "error": None,
        }
        result = FeedbackResult.from_dict(data)
        assert result.glyph_char == "c"
        assert result.ratings["aggressive"] == 8.0
        assert result.success is True

    def test_feedback_result_with_error(self):
        """Test feedback result with error."""
        from src.feedback import FeedbackResult

        result = FeedbackResult(
            glyph_char="x",
            ratings={},
            timestamp=datetime.now(),
            model="test-model",
            success=False,
            error="API timeout",
        )
        assert result.success is False
        assert result.error == "API timeout"
        assert result.ratings == {}


class TestEvaluationRun:
    """Tests for EvaluationRun dataclass."""

    def test_create_evaluation_run(self):
        """Test creating an evaluation run."""
        from src.feedback import EvaluationRun, FeedbackResult

        results = [
            FeedbackResult(
                glyph_char="a",
                ratings={"fruity": 7.0},
                timestamp=datetime.now(),
                model="test-model",
                success=True,
            )
        ]
        run = EvaluationRun(
            run_number=1,
            font_id="1-fruity",
            personality="fruity",
            results=results,
        )
        assert run.run_number == 1
        assert run.font_id == "1-fruity"
        assert len(run.results) == 1

    def test_evaluation_run_serialization(self):
        """Test serializing and deserializing evaluation run."""
        from src.feedback import EvaluationRun, FeedbackResult

        timestamp = datetime.now()
        results = [
            FeedbackResult(
                glyph_char="a",
                ratings={"fruity": 7.0},
                timestamp=timestamp,
                model="test-model",
                success=True,
            )
        ]
        run = EvaluationRun(
            run_number=1,
            font_id="1-fruity",
            personality="fruity",
            results=results,
            timestamp=timestamp,
        )

        # Round-trip
        d = run.to_dict()
        run2 = EvaluationRun.from_dict(d)

        assert run2.run_number == run.run_number
        assert run2.font_id == run.font_id
        assert len(run2.results) == len(run.results)
        assert run2.results[0].glyph_char == "a"


class TestResponseParsing:
    """Tests for Claude response parsing."""

    def test_parse_simple_json(self, mock_anthropic):
        """Test parsing simple JSON response."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            ratings = client._parse_response('{"fruity": 7, "aggressive": 3}')
            assert ratings["fruity"] == 7.0
            assert ratings["aggressive"] == 3.0

    def test_parse_json_in_code_block(self, mock_anthropic):
        """Test parsing JSON within markdown code block."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            response = '```json\n{"fruity": 8, "elegant": 5}\n```'
            ratings = client._parse_response(response)
            assert ratings["fruity"] == 8.0
            assert ratings["elegant"] == 5.0

    def test_parse_json_with_surrounding_text(self, mock_anthropic):
        """Test parsing JSON with surrounding text."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            response = 'Here are the ratings: {"fruity": 6, "dumb": 4} based on my analysis.'
            ratings = client._parse_response(response)
            assert ratings["fruity"] == 6.0
            assert ratings["dumb"] == 4.0

    def test_parse_clamps_out_of_range(self, mock_anthropic):
        """Test that out-of-range ratings are clamped."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            ratings = client._parse_response('{"fruity": 15, "aggressive": -5}')
            assert ratings["fruity"] == 10.0  # Clamped to max
            assert ratings["aggressive"] == 1.0  # Clamped to min

    def test_parse_normalizes_keys(self, mock_anthropic):
        """Test that keys are normalized to lowercase."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            ratings = client._parse_response('{"FRUITY": 7, "Aggressive": 3}')
            assert "fruity" in ratings
            assert "aggressive" in ratings

    def test_parse_invalid_json_raises(self, mock_anthropic):
        """Test that invalid JSON raises ValueError."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            with pytest.raises(ValueError, match="Failed to parse"):
                client._parse_response("This is not JSON at all")

    def test_parse_empty_ratings_raises(self, mock_anthropic):
        """Test that empty ratings raises ValueError."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            with pytest.raises(ValueError, match="No valid ratings"):
                client._parse_response('{"invalid": "not a number"}')


class TestClaudeFeedbackClient:
    """Tests for ClaudeFeedbackClient."""

    def test_client_requires_api_key(self):
        """Test that client requires API key."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if it exists
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ClaudeFeedbackClient()

    def test_client_accepts_api_key_param(self, mock_anthropic):
        """Test that client accepts API key as parameter."""
        from src.feedback import ClaudeFeedbackClient

        client = ClaudeFeedbackClient(api_key="test-key-param")
        assert client.api_key == "test-key-param"

    def test_evaluate_glyph_success(self, mock_anthropic, sample_glyph):
        """Test successful glyph evaluation."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            result = client.evaluate_glyph(sample_glyph, "a", "fruity")

            assert result.success is True
            assert result.glyph_char == "a"
            assert "fruity" in result.ratings
            assert result.ratings["fruity"] == 7.0

    def test_evaluate_glyph_encoding_error(self, mock_anthropic):
        """Test handling of encoding errors."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            # Pass invalid tensor (wrong shape)
            bad_tensor = torch.rand(3, 128, 128)  # RGB instead of grayscale
            result = client.evaluate_glyph(bad_tensor, "a", "fruity")

            assert result.success is False
            assert "Encoding error" in result.error

    def test_evaluate_glyph_api_error_retry(self, mock_anthropic, sample_glyph):
        """Test retry logic on API errors."""
        from src.feedback import ClaudeFeedbackClient

        # Make first call fail, second succeed
        mock_client = mock_anthropic.Anthropic.return_value
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"fruity": 7}')]
        mock_client.messages.create.side_effect = [Exception("Timeout"), mock_response]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.feedback.time.sleep"):  # Don't actually sleep in tests
                client = ClaudeFeedbackClient()
                result = client.evaluate_glyph(sample_glyph, "a", "fruity")

                assert result.success is True
                assert mock_client.messages.create.call_count == 2

    def test_evaluate_glyph_auth_error_no_retry(self, mock_anthropic, sample_glyph):
        """Test that auth errors don't retry."""
        from src.feedback import ClaudeFeedbackClient

        mock_client = mock_anthropic.Anthropic.return_value
        mock_client.messages.create.side_effect = Exception("AuthenticationError: 401")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            result = client.evaluate_glyph(sample_glyph, "a", "fruity")

            assert result.success is False
            assert "Authentication" in result.error
            assert mock_client.messages.create.call_count == 1  # No retry

    def test_evaluate_batch(self, mock_anthropic, sample_glyphs):
        """Test batch evaluation."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.feedback.time.sleep"):  # Don't actually sleep
                client = ClaudeFeedbackClient()
                glyphs = [(sample_glyphs["a"], "a"), (sample_glyphs["b"], "b")]
                results = client.evaluate_batch(glyphs, "fruity")

                assert len(results) == 2
                assert all(r.success for r in results)
                assert results[0].glyph_char == "a"
                assert results[1].glyph_char == "b"


class TestSampling:
    """Tests for glyph sampling."""

    def test_sample_glyphs_correct_size(self, sample_glyphs):
        """Test sampling correct number of glyphs."""
        from src.feedback import sample_glyphs as sample_fn

        sampled = sample_fn(sample_glyphs, sample_size=3)
        assert len(sampled) == 3

    def test_sample_glyphs_no_duplicates(self, sample_glyphs):
        """Test that sampled glyphs have no duplicates."""
        from src.feedback import sample_glyphs as sample_fn

        sampled = sample_fn(sample_glyphs, sample_size=5)
        chars = [char for _, char in sampled]
        assert len(chars) == len(set(chars))

    def test_sample_glyphs_deterministic_with_seed(self, sample_glyphs):
        """Test that sampling is deterministic with seed."""
        from src.feedback import sample_glyphs as sample_fn

        sampled1 = sample_fn(sample_glyphs, sample_size=3, seed=42)
        sampled2 = sample_fn(sample_glyphs, sample_size=3, seed=42)

        chars1 = [char for _, char in sampled1]
        chars2 = [char for _, char in sampled2]
        assert chars1 == chars2

    def test_sample_glyphs_small_population(self):
        """Test sampling when population is smaller than sample size."""
        from src.feedback import sample_glyphs as sample_fn

        small_glyphs = {"a": torch.rand(1, 128, 128), "b": torch.rand(1, 128, 128)}
        sampled = sample_fn(small_glyphs, sample_size=5)

        # Should only return what's available
        assert len(sampled) == 2


class TestFeedbackStorage:
    """Tests for feedback file storage."""

    def test_get_feedback_dir(self, temp_networks_dir):
        """Test getting feedback directory path."""
        from src.feedback import get_feedback_dir

        path = get_feedback_dir("1-fruity")
        assert path == temp_networks_dir / "1-fruity" / "feedback"

    def test_get_feedback_file_path(self, temp_networks_dir):
        """Test getting feedback file path."""
        from src.feedback import get_feedback_file_path

        path = get_feedback_file_path("1-fruity", 1, 0, "a")
        expected = temp_networks_dir / "1-fruity" / "feedback" / "1_eval_0_glyph-a_feedback.txt"
        assert path == expected

    def test_save_feedback(self, temp_networks_dir):
        """Test saving feedback to file."""
        from src.feedback import FeedbackResult, save_feedback

        result = FeedbackResult(
            glyph_char="a",
            ratings={"fruity": 7.0},
            timestamp=datetime.now(),
            model="test-model",
            success=True,
        )

        path = save_feedback("1-fruity", 1, 0, result)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["glyph_char"] == "a"
        assert data["ratings"]["fruity"] == 7.0

    def test_save_evaluation_run(self, temp_networks_dir):
        """Test saving evaluation run."""
        from src.feedback import EvaluationRun, FeedbackResult, save_evaluation_run

        results = [
            FeedbackResult(
                glyph_char="a",
                ratings={"fruity": 7.0},
                timestamp=datetime.now(),
                model="test-model",
                success=True,
            )
        ]
        run = EvaluationRun(
            run_number=1,
            font_id="1-fruity",
            personality="fruity",
            results=results,
        )

        path = save_evaluation_run("1-fruity", run)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["run_number"] == 1
        assert len(data["results"]) == 1

    def test_load_evaluation_run(self, temp_networks_dir):
        """Test loading evaluation run."""
        from src.feedback import (
            EvaluationRun,
            FeedbackResult,
            load_evaluation_run,
            save_evaluation_run,
        )

        results = [
            FeedbackResult(
                glyph_char="b",
                ratings={"aggressive": 8.0},
                timestamp=datetime.now(),
                model="test-model",
                success=True,
            )
        ]
        run = EvaluationRun(
            run_number=2,
            font_id="1-fruity",
            personality="fruity",
            results=results,
        )
        save_evaluation_run("1-fruity", run)

        loaded = load_evaluation_run("1-fruity", 2)

        assert loaded is not None
        assert loaded.run_number == 2
        assert loaded.results[0].glyph_char == "b"

    def test_load_nonexistent_run(self, temp_networks_dir):
        """Test loading nonexistent run returns None."""
        from src.feedback import load_evaluation_run

        loaded = load_evaluation_run("1-fruity", 999)
        assert loaded is None

    def test_get_next_run_number_empty(self, temp_networks_dir):
        """Test getting next run number when no runs exist."""
        from src.feedback import get_next_run_number

        next_num = get_next_run_number("1-fruity")
        assert next_num == 1

    def test_get_next_run_number_with_existing(self, temp_networks_dir):
        """Test getting next run number with existing runs."""
        from src.feedback import (
            EvaluationRun,
            get_next_run_number,
            save_evaluation_run,
        )

        for i in [1, 3, 5]:
            run = EvaluationRun(
                run_number=i,
                font_id="1-fruity",
                personality="fruity",
            )
            save_evaluation_run("1-fruity", run)

        next_num = get_next_run_number("1-fruity")
        assert next_num == 6

    def test_list_evaluation_runs(self, temp_networks_dir):
        """Test listing all evaluation runs."""
        from src.feedback import (
            EvaluationRun,
            list_evaluation_runs,
            save_evaluation_run,
        )

        for i in [1, 3, 2]:
            run = EvaluationRun(
                run_number=i,
                font_id="1-fruity",
                personality="fruity",
            )
            save_evaluation_run("1-fruity", run)

        runs = list_evaluation_runs("1-fruity")
        assert runs == [1, 2, 3]  # Should be sorted


class TestCollectFeedback:
    """Tests for the main collect_feedback function."""

    def test_collect_feedback_success(self, mock_anthropic, temp_networks_dir, sample_glyphs):
        """Test successful feedback collection."""
        from src.feedback import collect_feedback

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.feedback.time.sleep"):
                run = collect_feedback(
                    font_id="1-fruity",
                    glyphs=sample_glyphs,
                    personality="fruity",
                    sample_size=2,
                    seed=42,
                )

                assert run.run_number == 1
                assert run.font_id == "1-fruity"
                assert len(run.results) == 2
                assert all(r.success for r in run.results)

    def test_collect_feedback_no_api_key(self, temp_networks_dir, sample_glyphs):
        """Test feedback collection without API key returns empty run."""
        from src.feedback import collect_feedback

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            run = collect_feedback(
                font_id="1-fruity",
                glyphs=sample_glyphs,
                personality="fruity",
            )

            # Should return empty run without crashing
            assert run.run_number == 1
            assert len(run.results) == 0

    def test_collect_feedback_saves_files(self, mock_anthropic, temp_networks_dir, sample_glyphs):
        """Test that feedback collection saves all expected files."""
        from src.feedback import collect_feedback, get_feedback_dir

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.feedback.time.sleep"):
                run = collect_feedback(
                    font_id="1-fruity",
                    glyphs=sample_glyphs,
                    personality="fruity",
                    sample_size=2,
                    seed=42,
                )

                feedback_dir = get_feedback_dir("1-fruity")

                # Check summary file exists
                summary_file = feedback_dir / "run_1_summary.json"
                assert summary_file.exists()

                # Check individual feedback files exist
                individual_files = list(feedback_dir.glob("*_eval_*_glyph-*_feedback.txt"))
                assert len(individual_files) == 2


class TestComputeAverageRatings:
    """Tests for computing average ratings."""

    def test_compute_average_single_run(self):
        """Test computing average from single run."""
        from src.feedback import EvaluationRun, FeedbackResult, compute_average_ratings

        results = [
            FeedbackResult(
                glyph_char="a",
                ratings={"fruity": 6.0, "aggressive": 4.0},
                timestamp=datetime.now(),
                model="test",
                success=True,
            ),
            FeedbackResult(
                glyph_char="b",
                ratings={"fruity": 8.0, "aggressive": 2.0},
                timestamp=datetime.now(),
                model="test",
                success=True,
            ),
        ]
        run = EvaluationRun(
            run_number=1,
            font_id="1-fruity",
            personality="fruity",
            results=results,
        )

        averages = compute_average_ratings([run])
        assert averages["fruity"] == 7.0  # (6 + 8) / 2
        assert averages["aggressive"] == 3.0  # (4 + 2) / 2

    def test_compute_average_excludes_failed(self):
        """Test that failed evaluations are excluded from averages."""
        from src.feedback import EvaluationRun, FeedbackResult, compute_average_ratings

        results = [
            FeedbackResult(
                glyph_char="a",
                ratings={"fruity": 6.0},
                timestamp=datetime.now(),
                model="test",
                success=True,
            ),
            FeedbackResult(
                glyph_char="b",
                ratings={},
                timestamp=datetime.now(),
                model="test",
                success=False,
                error="API error",
            ),
        ]
        run = EvaluationRun(
            run_number=1,
            font_id="1-fruity",
            personality="fruity",
            results=results,
        )

        averages = compute_average_ratings([run])
        assert averages["fruity"] == 6.0  # Only one successful result

    def test_compute_average_multiple_runs(self):
        """Test computing average across multiple runs."""
        from src.feedback import EvaluationRun, FeedbackResult, compute_average_ratings

        runs = []
        for i in range(3):
            results = [
                FeedbackResult(
                    glyph_char="a",
                    ratings={"fruity": float(4 + i * 2)},  # 4, 6, 8
                    timestamp=datetime.now(),
                    model="test",
                    success=True,
                )
            ]
            runs.append(
                EvaluationRun(
                    run_number=i + 1,
                    font_id="1-fruity",
                    personality="fruity",
                    results=results,
                )
            )

        averages = compute_average_ratings(runs)
        assert averages["fruity"] == 6.0  # (4 + 6 + 8) / 3


class TestPromptBuilding:
    """Tests for prompt engineering."""

    def test_build_prompt_includes_personality(self, mock_anthropic):
        """Test that prompt includes target personality."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            prompt = client._build_prompt("fruity")

            assert "fruity" in prompt.lower()

    def test_build_prompt_includes_all_dimensions(self, mock_anthropic):
        """Test that prompt includes all default dimensions."""
        from src.feedback import PERSONALITY_PROMPTS, ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            prompt = client._build_prompt("fruity")

            for dim in PERSONALITY_PROMPTS:
                assert dim.upper() in prompt

    def test_build_prompt_custom_dimensions(self, mock_anthropic):
        """Test prompt with custom dimensions."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            prompt = client._build_prompt("fruity", dimensions=["fruity", "elegant"])

            assert "FRUITY" in prompt
            assert "ELEGANT" in prompt
            assert "AGGRESSIVE" not in prompt

    def test_build_prompt_requests_json(self, mock_anthropic):
        """Test that prompt requests JSON format."""
        from src.feedback import ClaudeFeedbackClient

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = ClaudeFeedbackClient()
            prompt = client._build_prompt("fruity")

            assert "JSON" in prompt

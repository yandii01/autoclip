"""Processing only part of a source: timecode parsing, time shifting, API validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from autoclip.pipeline import prepare
from autoclip.pipeline.transcript import Segment, Transcript, Word
from autoclip.timecode import parse_timecode


class TestParseTimecode:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("", 0.0),
            ("   ", 0.0),
            ("90", 90.0),
            ("1:30", 90.0),
            ("01:00:00", 3600.0),
            ("1:02:03", 3723.0),
            ("0:00:10.5", 10.5),
        ],
    )
    def test_valid(self, text: str, seconds: float) -> None:
        assert parse_timecode(text) == seconds

    def test_none_and_numbers(self) -> None:
        assert parse_timecode(None) == 0.0
        assert parse_timecode(42) == 42.0

    @pytest.mark.parametrize("text", ["abc", "1:2:3:4", "1:75", "-5", "1::30", "1:60:00"])
    def test_invalid(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_timecode(text)


class TestTranscriptShift:
    def test_shifts_words_and_segments(self) -> None:
        transcript = Transcript(
            words=[Word("a", 0.0, 0.5), Word("b", 1.0, 1.5)],
            segments=[Segment("a b", 0.0, 1.5, 0, 2)],
        )
        transcript.shift(3600.0)
        assert [(w.start, w.end) for w in transcript.words] == [(3600.0, 3600.5), (3601.0, 3601.5)]
        assert (transcript.segments[0].start, transcript.segments[0].end) == (3600.0, 3601.5)

    def test_zero_offset_is_a_no_op(self) -> None:
        transcript = Transcript(words=[Word("a", 1.0, 2.0)])
        transcript.shift(0.0)
        assert transcript.words[0].start == 1.0

    def test_index_lookup_uses_source_time_after_shift(self) -> None:
        transcript = Transcript(words=[Word("a", 0.0, 0.5), Word("b", 1.0, 1.5)])
        transcript.shift(100.0)
        assert transcript.index_at_time(101.2) == 1


def _tone(path: Path, seconds: int) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def _duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return float(out.stdout.strip())


@pytest.mark.slow
class TestExtractAudioRange:
    def test_whole_file_by_default(self, tmp_path: Path) -> None:
        source = _tone(tmp_path / "tone.wav", 10)
        out = prepare.extract_audio(source, tmp_path / "out.wav")
        assert _duration(out) == pytest.approx(10.0, abs=0.1)

    def test_start_and_end_select_a_slice(self, tmp_path: Path) -> None:
        source = _tone(tmp_path / "tone.wav", 10)
        out = prepare.extract_audio(source, tmp_path / "out.wav", start_s=3.0, end_s=7.0)
        assert _duration(out) == pytest.approx(4.0, abs=0.1)

    def test_start_only_runs_to_the_end(self, tmp_path: Path) -> None:
        source = _tone(tmp_path / "tone.wav", 10)
        out = prepare.extract_audio(source, tmp_path / "out.wav", start_s=6.0)
        assert _duration(out) == pytest.approx(4.0, abs=0.1)


class TestRunnerRange:
    def _runner(self, *, duration: float, start: float = 0.0, end: float = 0.0):
        from autoclip import config
        from autoclip.db.models import Job, Source, new_id
        from autoclip.pipeline.runner import PipelineRunner

        settings = config.Settings()
        settings.clips.range_start_s = start
        settings.clips.range_end_s = end
        source = Source(id=new_id(), type="upload", path="x.mp4", duration_s=duration)
        return PipelineRunner(Job(id=new_id(), source_id=source.id), source, settings=settings)

    def test_default_is_the_whole_source(self) -> None:
        assert self._runner(duration=600)._range() == (0.0, 600.0)

    def test_end_past_the_duration_is_clamped(self) -> None:
        assert self._runner(duration=600, start=60, end=9999)._range() == (60.0, 600.0)

    def test_explicit_slice(self) -> None:
        runner = self._runner(duration=14400, start=3600, end=7200)
        assert runner._range() == (3600.0, 7200.0)
        assert runner._range_length() == 3600.0

    def test_start_past_the_end_fails_clearly(self) -> None:
        from autoclip.pipeline.runner import PipelineError

        with pytest.raises(PipelineError, match="past the end"):
            self._runner(duration=600, start=700)._range()

    def test_start_after_end_fails(self) -> None:
        from autoclip.pipeline.runner import PipelineError

        with pytest.raises(PipelineError, match="before"):
            self._runner(duration=600, start=300, end=100)._range()

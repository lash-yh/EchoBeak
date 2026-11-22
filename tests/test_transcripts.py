import pytest

from utils import paths as output_paths
from utils import transcripts


def test_default_transcript_directory_and_dated_name(tmp_path, monkeypatch):
    directory = tmp_path / "EchoBeak-Transcription"
    monkeypatch.setattr(output_paths, "_ECHOBEAK_DIR", directory)

    output = transcripts.create_name_trans_path()

    assert output == directory / f"trans-{transcripts.date.today():%Y-%m-%d}.md"
    assert directory.is_dir()


def test_publish_preserves_existing_transcript(tmp_path):
    output = tmp_path / "transcript.md"
    output.write_text("existing", encoding="utf-8")
    draft = tmp_path / "draft.md"
    draft.write_text("voz con acentos: canción", encoding="utf-8")
    published = transcripts.publish_transcript(draft, output)
    assert published.name == "transcript-(1).md"
    assert published.read_text(encoding="utf-8") == draft.read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("symbolic_link", [False, True])
def test_publish_retries_concurrent_collision(tmp_path, monkeypatch, symbolic_link):
    output = tmp_path / "transcript.md"
    draft = tmp_path / "draft.md"
    draft.write_text("new", encoding="utf-8")
    unique_path = transcripts._unique_path
    calls = 0

    def racing_path(path):
        nonlocal calls
        candidate = unique_path(path)
        if calls == 0:
            if symbolic_link:
                candidate.symlink_to(tmp_path / "missing.md")
            else:
                candidate.write_text("concurrent", encoding="utf-8")
        calls += 1
        assert calls <= 2, "Publication repeatedly selected an occupied path"
        return candidate

    monkeypatch.setattr(transcripts, "_unique_path", racing_path)
    published = transcripts.publish_transcript(draft, output)
    assert published.read_text(encoding="utf-8") == "new"
    if symbolic_link:
        assert output.is_symlink()
        assert not output.exists()
    else:
        assert output.read_text(encoding="utf-8") == "concurrent"
    assert calls == 2


@pytest.mark.parametrize("target_exists", [False, True])
def test_publish_skips_symbolic_links_in_base_and_suffix(tmp_path, target_exists):
    output = tmp_path / "transcript.md"
    target = tmp_path / "target.md"
    if target_exists:
        target.write_text("original", encoding="utf-8")
    links = [output, tmp_path / "transcript-(1).md"]
    for link in links:
        link.symlink_to(target)
    draft = tmp_path / "draft.md"
    draft.write_text("new transcript", encoding="utf-8")
    expected = tmp_path / "transcript-(2).md"

    assert transcripts._unique_path(output) == expected
    published = transcripts.publish_transcript(draft, output)

    assert published == expected
    assert published.read_text(encoding="utf-8") == "new transcript"
    assert all(link.is_symlink() and link.readlink() == target for link in links)
    if target_exists:
        assert target.read_text(encoding="utf-8") == "original"
    else:
        assert not target.exists()


def test_unique_path_stops_when_broken_symlinks_exhaust_candidates(tmp_path):
    output = tmp_path / "transcript.md"
    for name in ["transcript.md", "transcript-(1).md", "transcript-(2).md"]:
        (tmp_path / name).symlink_to(tmp_path / "missing.md")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        transcripts._unique_path(output, max_attempts=2)


def test_publish_failure_removes_partial_file(tmp_path, monkeypatch):
    output = tmp_path / "transcript.md"
    draft = tmp_path / "draft.md"
    draft.write_text("new", encoding="utf-8")
    failure = OSError("disk full")

    def fail(source, destination):
        destination.write("partial")
        raise failure

    monkeypatch.setattr(transcripts.shutil, "copyfileobj", fail)
    with pytest.raises(OSError) as caught:
        transcripts.publish_transcript(draft, output)
    assert caught.value is failure
    assert not output.exists()

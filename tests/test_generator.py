"""Tests for comp_starter.generator module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from comp_starter.generator import list_custom_templates, submit_file


class TestListCustomTemplates:
    """Tests for list_custom_templates()."""

    def test_returns_list(self):
        result = list_custom_templates()
        assert isinstance(result, list)

    def test_items_have_required_keys(self):
        results = list_custom_templates()
        for item in results:
            assert "name" in item
            assert "path" in item
            assert "description" in item
            assert isinstance(item["name"], str)
            assert isinstance(item["path"], str)
            assert isinstance(item["description"], str)


class TestSubmitFile:
    """Tests for submit_file()."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.submissions_dir = Path(self.tmpdir) / "submissions"
        self.submissions_dir.mkdir()
        self.src_file = self.submissions_dir / "submission.csv"
        self.src_file.write_text("id,pred\n1,0.5\n")

    def teardown_method(self):
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_raises_if_source_missing(self):
        with pytest.raises(FileNotFoundError):
            submit_file("/nonexistent/file.csv")

    def test_copies_file_to_submissions(self):
        dest = submit_file(str(self.src_file))
        assert dest.exists()
        assert dest.name.startswith(self.src_file.stem)
        assert dest.suffix == ".csv"

    def test_creates_metadata_file(self):
        dest = submit_file(str(self.src_file), note="first submission")
        stem = dest.stem
        meta_path = dest.parent / f"{stem}_meta.json"
        assert meta_path.exists()

    def test_metadata_contains_expected_fields(self):
        dest = submit_file(str(self.src_file), note="test note")
        stem = dest.stem
        meta_path = dest.parent / f"{stem}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert "version" in meta
        assert "source" in meta
        assert "note" in meta
        assert "timestamp" in meta
        assert meta["note"] == "test note"

    def test_version_increments(self):
        dest1 = submit_file(str(self.src_file))
        dest2 = submit_file(str(self.src_file))
        v1 = int(dest1.stem.split("_v")[-1])
        v2 = int(dest2.stem.split("_v")[-1])
        assert v2 > v1

    def test_returns_path_object(self):
        dest = submit_file(str(self.src_file))
        assert isinstance(dest, Path)
"""P6 annotations: a 1-line-commit YAML file of dated notes."""
import datetime as dt

import pytest

from src.analytics.annotations import load_annotations


def test_missing_file_is_empty(tmp_path):
    df = load_annotations(tmp_path / "annotations.yaml")
    assert df.empty and list(df.columns) == ["date", "note"]


def test_empty_list_is_empty(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("annotations: []\n")
    assert load_annotations(p).empty


def test_entries_parse_to_dates(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("annotations:\n  - {date: 2026-09-17, note: FOMC}\n  - date: '2026-10-02'\n    note: payrolls\n")
    df = load_annotations(p)
    assert list(df["date"]) == [dt.date(2026, 9, 17), dt.date(2026, 10, 2)]
    assert list(df["note"]) == ["FOMC", "payrolls"]


def test_bad_entry_raises(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("annotations:\n  - {date: 2026-09-17}\n")
    with pytest.raises(ValueError):
        load_annotations(p)


def test_shipped_file_loads():
    from pathlib import Path
    df = load_annotations(Path("data/annotations.yaml"))
    assert list(df.columns) == ["date", "note"]

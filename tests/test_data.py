import json

import pytest

from injection_guard import data
from injection_guard.data import Example, read_jsonl, stratified_split, text_hash


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_read_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    _write(p, [{"id": "a", "text": "hi", "label": 0}, {"id": "b", "text": "ignore", "label": 1}])
    ex = read_jsonl(p)
    assert ex == [Example("a", "hi", 0), Example("b", "ignore", 1)]


def test_read_jsonl_rejects_bad_label(tmp_path):
    p = tmp_path / "x.jsonl"
    _write(p, [{"id": "a", "text": "hi", "label": 2}])
    with pytest.raises(ValueError):
        read_jsonl(p)


def test_split_is_deterministic_stratified_and_disjoint():
    ex = [Example(f"t-{i}", f"text {i}", int(i % 3 == 0)) for i in range(200)]
    tr1, va1 = stratified_split(ex, 0.15, seed=42)
    tr2, va2 = stratified_split(ex, 0.15, seed=42)
    assert tr1 == tr2 and va1 == va2
    assert {e.id for e in tr1}.isdisjoint({e.id for e in va1})
    assert len(tr1) + len(va1) == len(ex)
    pos_rate = sum(e.label for e in ex) / len(ex)
    val_rate = sum(e.label for e in va1) / len(va1)
    assert abs(pos_rate - val_rate) < 0.05


def test_text_hash_stable():
    assert text_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_load_splits_from_dir(tmp_path):
    _write(tmp_path / "train.jsonl", [{"id": f"train-{i}", "text": str(i), "label": i % 2} for i in range(40)])
    _write(tmp_path / "test.jsonl", [{"id": f"test-{i}", "text": str(i), "label": i % 2} for i in range(10)])
    s = data.load_splits(tmp_path)
    assert len(s["test"]) == 10
    assert len(s["train"]) + len(s["val"]) == 40


def test_load_splits_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_splits(tmp_path)


@pytest.mark.skipif(not (data.DATA_DIR / "train.jsonl").exists(), reason="dataset not downloaded")
def test_real_dataset_shape():
    s = data.load_splits()
    assert len(s["train"]) + len(s["val"]) == 546
    assert len(s["test"]) == 116
    ids = [e.id for k in s for e in s[k]]
    assert len(ids) == len(set(ids))

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from glm53_nvfp4.audit_p8_fullmodel_completion import verify_dispatch, verify_records
from glm53_nvfp4.shard_index import sha256_file


def _log():
    lines = []
    for layer in range(3, 45):
        for rank in range(4):
            common = f"layer={layer} rank={rank} stream=K4 alphabet=E4M3 scale=UE8M0_K32 boundary=identity ldlq=false"
            lines.append(f"GLM53_P8_NATIVE_WEIGHTS_READY {common} design_sha256=abc law=mcg")
            lines.append(f"GLM53_P8_NATIVE_FORWARD {common} mma=mxf8f6f4 law=procedural_mcg deterministic=route_topk_sum physical_bpw=4.25")
    return "\n".join(lines)


def test_dispatch_requires_every_rank_and_native_contract():
    text = _log()
    assert verify_dispatch(text, "abc") == {"weights_ready": 168, "forward": 168}
    with pytest.raises(ValueError, match="inventory"):
        verify_dispatch("\n".join(text.splitlines()[:-1]), "abc")
    with pytest.raises(ValueError, match="contract"):
        verify_dispatch(text.replace("mma=mxf8f6f4", "mma=bf16", 1), "abc")
    with pytest.raises(ValueError, match="contract"):
        verify_dispatch(text, "wrong-design")


def _fixture(root):
    rows = [{"id": f"w{i}", "domain": f"d{i//8}", "prediction_positions": 3, "teacher_sha256": "teacher"} for i in range(32)]
    manifest = {"status": "complete", "role": "conditional-fit", "roles_sha256": "roles", "run_id": "test", "config_id": "native", "windows": {}}
    for row in rows:
        path = root / (row["id"]+".parquet")
        table = pa.table({"window_id": [row["id"]]*3, "run_id": ["test"]*3, "config_id": ["native"]*3, "domain": [row["domain"]]*3, "context_len": [4]*3, "pos": [0,1,2], "kld": [0.1,0.2,0.3]}).replace_schema_metadata({"role": "conditional-fit", "roles_sha256": "roles", "teacher_sha256": "teacher", "student_logits_sha256": "student"})
        pq.write_table(table, path)
        manifest["windows"][row["id"]] = {"record_sha256": sha256_file(path), "record_path": str(path), "student_logits_sha256": "student", "mean_kld": 0.2}
    return rows, manifest


@pytest.mark.parametrize("fault", ["missing_position", "duplicate_position", "nan", "wrong_domain", "wrong_teacher", "receipt", "missing_window"])
def test_rejects_corrupt_record_even_if_window_average_could_exist(tmp_path, fault):
    rows, manifest = _fixture(tmp_path)
    assert len(verify_records(rows, tmp_path, manifest, "roles")) == 32
    path = tmp_path / "w0.parquet"
    table = pq.read_table(path)
    if fault == "missing_position":
        table = table.slice(0,2)
    elif fault == "duplicate_position":
        table = table.set_column(table.schema.get_field_index("pos"), "pos", pa.array([0,1,1]))
    elif fault == "nan":
        table = table.set_column(table.schema.get_field_index("kld"), "kld", pa.array([0.1,float("nan"),0.3]))
    elif fault == "wrong_domain":
        table = table.set_column(table.schema.get_field_index("domain"), "domain", pa.array(["d9"]*3))
    elif fault == "wrong_teacher":
        table = table.replace_schema_metadata({**table.schema.metadata, b"teacher_sha256": b"wrong"})
    elif fault == "receipt":
        manifest["windows"]["w0"]["record_sha256"] = "wrong"
    elif fault == "missing_window":
        path.unlink()
    if fault not in ("receipt", "missing_window"):
        pq.write_table(table, path)
        manifest["windows"]["w0"]["record_sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        verify_records(rows, tmp_path, manifest, "roles")

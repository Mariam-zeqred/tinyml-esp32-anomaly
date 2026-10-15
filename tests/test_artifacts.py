"""The committed model, results and generated C headers must agree with each other."""

import json
import re

from tinyml.load import ROOT


def define(text: str, name: str) -> str:
    m = re.search(rf"#define {name} (\S+)", text)
    assert m, name
    return m.group(1).rstrip("f")


def test_params_header_matches_results_and_model():
    params = (ROOT / "lib/anomaly_core/model_params.h").read_text()
    results = json.loads((ROOT / "figures/results.json").read_text())
    report = json.loads((ROOT / "model/train_report.json").read_text())
    tflite = (ROOT / "model/anomaly_ae_int8.tflite").read_bytes()
    assert tflite[4:8] == b"TFL3"
    assert int(define(params, "AN_MODEL_BYTES")) == len(tflite) == results["tflite_bytes"]
    assert int(define(params, "AN_N_PARAMS")) == report["parameters"]
    assert abs(float(define(params, "AN_THRESHOLD")) - results["threshold"]["value"]) < 1e-9
    assert int(define(params, "AN_WINDOW")) == 32


def test_model_data_cc_is_the_tflite_file():
    cc = (ROOT / "lib/anomaly_core/model_data.cc").read_text()
    tflite = (ROOT / "model/anomaly_ae_int8.tflite").read_bytes()
    body = cc.split("{", 1)[1].split("}", 1)[0]
    data = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", body))
    assert data == tflite

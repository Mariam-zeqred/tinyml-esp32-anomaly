"""tinyml-esp32-anomaly: a tiny autoencoder that scores 8-minute incubator windows.

Pipeline (all host side): ``load`` the real trace -> ``windows`` (features + split) ->
``train`` (Keras autoencoder -> int8 TensorFlow Lite) -> ``export_c`` (C arrays for the
ESP32 sketch and the host test) -> ``evaluate`` (threshold, injected ``faults``, report).
"""

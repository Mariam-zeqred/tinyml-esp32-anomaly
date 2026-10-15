// Host-side reference of the autoencoder: the dequantised weights of the .tflite file run in
// float. Not used on the ESP32 (it runs the int8 model with TensorFlow Lite Micro); it exists
// so `make check` can prove, without the TFLM library, that model_data.cc really encodes the
// trained network and that the window code feeds it correctly.
#pragma once
#include "model_params.h"

void ref_model_infer(const float f[AN_N_FEATURES], float recon[AN_N_TARGETS]);

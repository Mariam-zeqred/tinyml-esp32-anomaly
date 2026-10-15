// TensorFlow Lite Micro wrapper around the int8 autoencoder in lib/anomaly_core/model_data.cc.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "model_params.h"

class Detector {
 public:
  // Load the model and allocate tensors in `arena`. Returns false (and prints why) on failure.
  bool begin(uint8_t* arena, size_t arena_bytes);
  // Reconstruct one window: features in, dequantised temperature reconstruction out.
  bool infer(const float f[AN_N_FEATURES], float recon[AN_N_TARGETS]);
  size_t arena_used() const;
  uint32_t last_infer_us() const { return last_us_; }

 private:
  uint32_t last_us_ = 0;
};

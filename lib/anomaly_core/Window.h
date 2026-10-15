// Ring buffer of the last AN_WINDOW readings and the feature vector the model consumes.
// Mirrors src/tinyml/windows.py exactly; the host test checks both against the same vectors.
#pragma once
#include <stdint.h>

#include "model_params.h"

class AnomalyWindow {
 public:
  AnomalyWindow() { reset(); }
  void reset();
  // Add one reading (every ~15 s). Returns true once AN_WINDOW readings are buffered.
  bool push(float temp_c, bool heater_on);
  bool full() const { return count_ >= AN_WINDOW; }
  // f[0..AN_WINDOW-2] = (temp[k]-temp[0]) / AN_DELTA_SCALE_C, f[AN_WINDOW-1..] = heater[k].
  // Requires full(). Returns false (and leaves f untouched) otherwise.
  bool features(float f[AN_N_FEATURES]) const;

 private:
  float temp_[AN_WINDOW];
  uint8_t heater_[AN_WINDOW];
  int head_;   // index of the oldest reading once full
  int count_;
};

// Quantise a feature vector for the int8 model input; dequantise the int8 output.
void an_quantize(const float f[AN_N_FEATURES], int8_t q[AN_N_FEATURES]);
void an_dequantize(const int8_t q[AN_N_TARGETS], float recon[AN_N_TARGETS]);
// Anomaly score: mean squared error between the temperature features and their reconstruction.
float an_score(const float f[AN_N_FEATURES], const float recon[AN_N_TARGETS]);

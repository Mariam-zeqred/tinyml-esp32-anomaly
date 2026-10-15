#include "Window.h"

#include <math.h>

void AnomalyWindow::reset() {
  head_ = 0;
  count_ = 0;
  for (int i = 0; i < AN_WINDOW; i++) {
    temp_[i] = 0.0f;
    heater_[i] = 0;
  }
}

bool AnomalyWindow::push(float temp_c, bool heater_on) {
  if (count_ < AN_WINDOW) {
    temp_[count_] = temp_c;
    heater_[count_] = heater_on ? 1 : 0;
    count_++;
    return count_ >= AN_WINDOW;
  }
  temp_[head_] = temp_c;  // overwrite the oldest, which then becomes the newest
  heater_[head_] = heater_on ? 1 : 0;
  head_ = (head_ + 1) % AN_WINDOW;
  return true;
}

bool AnomalyWindow::features(float f[AN_N_FEATURES]) const {
  if (!full()) return false;
  const float t0 = temp_[head_];
  for (int k = 1; k < AN_WINDOW; k++) {
    f[k - 1] = (temp_[(head_ + k) % AN_WINDOW] - t0) / AN_DELTA_SCALE_C;
  }
  for (int k = 0; k < AN_WINDOW; k++) {
    f[AN_WINDOW - 1 + k] = (float)heater_[(head_ + k) % AN_WINDOW];
  }
  return true;
}

void an_quantize(const float f[AN_N_FEATURES], int8_t q[AN_N_FEATURES]) {
  for (int i = 0; i < AN_N_FEATURES; i++) {
    float v = roundf(f[i] / AN_IN_SCALE) + (float)AN_IN_ZERO;
    if (v > 127.0f) v = 127.0f;
    if (v < -128.0f) v = -128.0f;
    q[i] = (int8_t)v;
  }
}

void an_dequantize(const int8_t q[AN_N_TARGETS], float recon[AN_N_TARGETS]) {
  for (int i = 0; i < AN_N_TARGETS; i++) {
    recon[i] = ((float)q[i] - (float)AN_OUT_ZERO) * AN_OUT_SCALE;
  }
}

float an_score(const float f[AN_N_FEATURES], const float recon[AN_N_TARGETS]) {
  float acc = 0.0f;
  for (int i = 0; i < AN_N_TARGETS; i++) {
    const float e = f[i] - recon[i];
    acc += e * e;
  }
  return acc / (float)AN_N_TARGETS;
}

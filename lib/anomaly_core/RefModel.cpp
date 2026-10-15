#ifdef NATIVE_TEST
#include "RefModel.h"

#include <math.h>

#include "Window.h"
#include "model_weights.h"

// y[out] = tanh?(sum_in W[out][in] * x[in] + b[out])
static void dense(const float* x, int n_in, const float* w, const float* b, int n_out,
                  bool use_tanh, float* y) {
  for (int o = 0; o < n_out; o++) {
    float acc = b[o];
    const float* row = w + o * n_in;
    for (int i = 0; i < n_in; i++) acc += row[i] * x[i];
    y[o] = use_tanh ? tanhf(acc) : acc;
  }
}

void ref_model_infer(const float f[AN_N_FEATURES], float recon[AN_N_TARGETS]) {
  // The int8 model rounds and clips its input; emulate that so out-of-range faults compare.
  int8_t q[AN_N_FEATURES];
  float x[AN_N_FEATURES], h1[AN_ENC_OUT], h2[AN_CODE_OUT], h3[AN_DEC_OUT];
  an_quantize(f, q);
  for (int i = 0; i < AN_N_FEATURES; i++) x[i] = ((float)q[i] - (float)AN_IN_ZERO) * AN_IN_SCALE;
  dense(x, AN_N_FEATURES, AN_W_ENC, AN_B_ENC, AN_ENC_OUT, true, h1);
  dense(h1, AN_ENC_OUT, AN_W_CODE, AN_B_CODE, AN_CODE_OUT, true, h2);
  dense(h2, AN_CODE_OUT, AN_W_DEC, AN_B_DEC, AN_DEC_OUT, true, h3);
  dense(h3, AN_DEC_OUT, AN_W_RECON, AN_B_RECON, AN_RECON_OUT, false, recon);
  // ... and its output can only hold what int8 represents.
  const float lo = (-128.0f - (float)AN_OUT_ZERO) * AN_OUT_SCALE;
  const float hi = (127.0f - (float)AN_OUT_ZERO) * AN_OUT_SCALE;
  for (int i = 0; i < AN_N_TARGETS; i++) recon[i] = fminf(hi, fmaxf(lo, recon[i]));
}
#endif  // NATIVE_TEST

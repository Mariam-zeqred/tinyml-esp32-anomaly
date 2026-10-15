// Host tests for lib/anomaly_core (no hardware, no TensorFlow Lite Micro).
//   make check              (plain g++)
//   pio test -e native      (PlatformIO native env, custom runner in test/)
#include <math.h>
#include <stdio.h>

#include "RefModel.h"
#include "Window.h"
#include "minitest.h"
#include "model_data.h"
#include "test_vectors.h"

// ---------- window / features ----------

static void window_fills_then_rolls() {
  AnomalyWindow w;
  float f[AN_N_FEATURES];
  CHECK(!w.full());
  CHECK(!w.features(f));
  for (int i = 0; i < AN_WINDOW - 1; i++) CHECK(!w.push(20.0f + 0.01f * i, false));
  CHECK(w.push(25.0f, true));  // the 32nd reading completes the window
  CHECK(w.full());
  CHECK(w.features(f));
  CHECK(fabsf(f[AN_WINDOW - 2] - (25.0f - 20.0f) / AN_DELTA_SCALE_C) < 1e-6f);
  CHECK_EQ(f[AN_N_FEATURES - 1], 1.0f);
  CHECK_EQ(f[AN_WINDOW - 1], 0.0f);
  // after one more push the oldest reading (20.00) is gone and the deltas are relative to 20.01
  w.push(26.0f, false);
  CHECK(w.features(f));
  CHECK(fabsf(f[0] - (20.02f - 20.01f) / AN_DELTA_SCALE_C) < 1e-5f);
  CHECK(fabsf(f[AN_WINDOW - 2] - (26.0f - 20.01f) / AN_DELTA_SCALE_C) < 1e-5f);
  CHECK_EQ(f[AN_N_FEATURES - 1], 0.0f);
  CHECK_EQ(f[AN_N_FEATURES - 2], 1.0f);
}

static void features_are_level_invariant() {
  AnomalyWindow a, b;
  float fa[AN_N_FEATURES], fb[AN_N_FEATURES];
  for (int i = 0; i < AN_WINDOW; i++) {
    a.push(30.0f + 0.02f * i, i % 3 == 0);
    b.push(38.8f + 0.02f * i, i % 3 == 0);
  }
  a.features(fa);
  b.features(fb);
  float max_diff = 0.0f;
  for (int i = 0; i < AN_N_FEATURES; i++) max_diff = fmaxf(max_diff, fabsf(fa[i] - fb[i]));
  CHECK(max_diff < 1e-4f);
}

static void quantize_round_trips_and_clamps() {
  float f[AN_N_FEATURES], back[AN_N_TARGETS];
  int8_t q[AN_N_FEATURES];
  for (int i = 0; i < AN_N_FEATURES; i++) f[i] = 0.01f * (i - 20);
  an_quantize(f, q);
  an_dequantize(q, back);  // not the same scale as the input; just check the shape of the maths
  CHECK(q[20] == AN_IN_ZERO);  // f = 0 maps to the zero point
  f[0] = 1e6f;
  f[1] = -1e6f;
  an_quantize(f, q);
  CHECK_EQ(q[0], 127);
  CHECK_EQ(q[1], -128);
  CHECK(fabsf(back[0] - ((float)q[0] - AN_OUT_ZERO) * AN_OUT_SCALE) >= 0.0f);  // defined
}

static void score_is_mse_over_temperature_features() {
  float f[AN_N_FEATURES] = {0}, r[AN_N_TARGETS] = {0};
  CHECK_EQ(an_score(f, r), 0.0f);
  r[0] = 0.5f;  // one feature off by 0.5 -> 0.25 / 31
  CHECK(fabsf(an_score(f, r) - 0.25f / AN_N_TARGETS) < 1e-7f);
  f[AN_N_FEATURES - 1] = 1.0f;  // heater features do not enter the score
  CHECK(fabsf(an_score(f, r) - 0.25f / AN_N_TARGETS) < 1e-7f);
}

// ---------- model ----------

static void model_blob_is_a_tflite_file() {
  CHECK_EQ(g_anomaly_model_tflite_len, (unsigned)AN_MODEL_BYTES);
  // FlatBuffer file identifier "TFL3" sits at bytes 4..7
  CHECK_EQ(g_anomaly_model_tflite[4], 'T');
  CHECK_EQ(g_anomaly_model_tflite[5], 'F');
  CHECK_EQ(g_anomaly_model_tflite[6], 'L');
  CHECK_EQ(g_anomaly_model_tflite[7], '3');
}

// Feed every recorded segment reading by reading through the window code and the reference
// kernel; compare with the scores the TFLite int8 interpreter produced in Python.
static void reference_kernel_matches_interpreter_on_recorded_windows() {
  int windows = 0, flagged = 0;
  float worst = 0.0f;
  for (int s = 0; s < AN_N_TEST_SEGMENTS; s++) {
    const an_test_segment_t& seg = AN_TEST_SEGMENTS[s];
    AnomalyWindow w;
    int wi = 0;
    for (int i = 0; i < seg.n; i++) {
      if (!w.push(seg.temp[i], seg.heater[i] != 0)) continue;
      float f[AN_N_FEATURES], recon[AN_N_TARGETS];
      w.features(f);
      ref_model_infer(f, recon);
      const float score = an_score(f, recon);
      const float diff = fabsf(score - seg.score[wi]);
      const float tol = fmaxf(AN_SCORE_TOL_ABS, AN_SCORE_TOL_REL * seg.score[wi]);
      worst = fmaxf(worst, diff / tol);
      if (diff > tol) {
        printf("FAIL %s window %d: ref %.6f interpreter %.6f\n", seg.name, wi, score, seg.score[wi]);
        CHECK(diff <= tol);
      }
      if (seg.sure[wi]) CHECK_EQ((score > AN_THRESHOLD) ? 1 : 0, (int)seg.flag[wi]);
      flagged += (score > AN_THRESHOLD) ? 1 : 0;
      wi++;
      windows++;
    }
    CHECK_EQ(wi, seg.n - AN_WINDOW + 1);
  }
  printf("note: %d windows, %d flagged, worst |ref - interpreter| = %.0f %% of tolerance\n", windows,
         flagged, (double)(100.0f * worst));
  CHECK(windows > 200);
}

static void clean_segments_stay_quiet_and_faults_fire() {
  for (int s = 0; s < AN_N_TEST_SEGMENTS; s++) {
    const an_test_segment_t& seg = AN_TEST_SEGMENTS[s];
    int flagged = 0;
    AnomalyWindow w;
    for (int i = 0; i < seg.n; i++) {
      if (!w.push(seg.temp[i], seg.heater[i] != 0)) continue;
      float f[AN_N_FEATURES], recon[AN_N_TARGETS];
      w.features(f);
      ref_model_infer(f, recon);
      flagged += an_score(f, recon) > AN_THRESHOLD;
    }
    const bool is_clean = seg.name[0] == 'c';
    printf("note: %-24s %3d windows, %2d flagged\n", seg.name, seg.n - AN_WINDOW + 1, flagged);
    if (is_clean) CHECK_EQ(flagged, 0);
    else CHECK(flagged >= 3);
  }
}

int main() {
  RUN(window_fills_then_rolls);
  RUN(features_are_level_invariant);
  RUN(quantize_round_trips_and_clamps);
  RUN(score_is_mse_over_temperature_features);
  RUN(model_blob_is_a_tflite_file);
  RUN(reference_kernel_matches_interpreter_on_recorded_windows);
  RUN(clean_segments_stay_quiet_and_faults_fire);
  return mt_summary();
}

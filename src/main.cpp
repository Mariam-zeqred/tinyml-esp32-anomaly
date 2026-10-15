// tinyml-esp32-anomaly: score 8-minute incubator temperature windows with an int8 autoencoder.
//
// Every SAMPLE_PERIOD_MS one reading (temperature, heater state) is pushed into the window;
// once 32 readings are buffered each new reading yields one score. Score > AN_THRESHOLD lights
// the LED and prints ALARM. With USE_REPLAY the recorded clean and faulted segments from the
// host test are played back, so the behaviour can be watched without a sensor.
#include <Arduino.h>

#include "Config.h"
#include "Detector.h"
#include "Replay.h"
#include "Window.h"
#include "model_params.h"
#include "test_vectors.h"

static Detector g_detector;
static AnomalyWindow g_window;
alignas(16) static uint8_t g_arena[TENSOR_ARENA_BYTES];

// Run the recorded segments through the model and compare with the host interpreter's scores.
static void self_check() {
  int windows = 0, bad = 0;
  float worst = 0.0f;
  for (int s = 0; s < AN_N_TEST_SEGMENTS; s++) {
    const an_test_segment_t& seg = AN_TEST_SEGMENTS[s];
    AnomalyWindow w;
    int wi = 0;
    for (int i = 0; i < seg.n; i++) {
      if (!w.push(seg.temp[i], seg.heater[i] != 0)) continue;
      float f[AN_N_FEATURES], recon[AN_N_TARGETS];
      w.features(f);
      if (!g_detector.infer(f, recon)) {
        Serial.println("self-check: Invoke failed");
        return;
      }
      const float score = an_score(f, recon);
      const float diff = fabsf(score - seg.score[wi]);
      const float tol = fmaxf(AN_SCORE_TOL_ABS, AN_SCORE_TOL_REL * seg.score[wi]);
      if (diff / tol > worst) worst = diff / tol;
      if (diff > tol) bad++;
      if (seg.sure[wi] && ((score > AN_THRESHOLD) ? 1 : 0) != seg.flag[wi]) bad++;
      wi++;
      windows++;
    }
  }
  Serial.printf("self-check: %s  %d windows, %d mismatches, worst gap %.0f %% of tolerance\n",
                bad == 0 ? "PASS" : "FAIL", windows, bad, (double)(100.0f * worst));
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  pinMode(LED_PIN, OUTPUT);
  Serial.printf("\ntinyml-esp32-anomaly  model %d bytes, %d parameters, window %d, threshold %.4g\n",
                AN_MODEL_BYTES, AN_N_PARAMS, AN_WINDOW, (double)AN_THRESHOLD);
  if (!g_detector.begin(g_arena, sizeof(g_arena))) {
    while (true) delay(1000);
  }
  Serial.printf("arena used %u of %u bytes\n", (unsigned)g_detector.arena_used(),
                (unsigned)TENSOR_ARENA_BYTES);
#if SELF_CHECK
  self_check();
  Serial.printf("inference %lu us\n", (unsigned long)g_detector.last_infer_us());
#endif
}

void loop() {
  static uint32_t last_ms = 0;
  const uint32_t period = USE_REPLAY ? REPLAY_PERIOD_MS : SAMPLE_PERIOD_MS;
  if (millis() - last_ms < period) return;
  last_ms = millis();

  Reading r;
  if (!read_next(r)) return;
  if (r.segment_start) {
    g_window.reset();
    Serial.printf("--- %s\n", r.segment);
  }
  if (!g_window.push(r.temp_c, r.heater_on)) return;  // still filling the first window

  float f[AN_N_FEATURES], recon[AN_N_TARGETS];
  g_window.features(f);
  if (!g_detector.infer(f, recon)) return;
  const float score = an_score(f, recon);
  const bool alarm = score > AN_THRESHOLD;
  digitalWrite(LED_PIN, alarm ? HIGH : LOW);
  Serial.printf("temp %.2f heater %d score %.5f %s (%lu us)\n", (double)r.temp_c, r.heater_on ? 1 : 0,
                (double)score, alarm ? "ALARM" : "ok", (unsigned long)g_detector.last_infer_us());
}

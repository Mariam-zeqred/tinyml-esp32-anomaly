// Tuning knobs for the sketch. Model constants (window, threshold, quantisation) are generated
// into lib/anomaly_core/model_params.h and must not be edited by hand.
#pragma once

#define SERIAL_BAUD 115200
#define LED_PIN 2                  // on-board LED of most ESP32 DevKit boards: on while a window is flagged

// Reading source. 1: replay the recorded segments from lib/anomaly_core/test_vectors.h (no
// sensor needed, what you see after flashing). 0: your own sensor, see src/Replay.cpp.
#define USE_REPLAY 1
#define SAMPLE_PERIOD_MS 15000     // the incubator samples every 15 s ...
#define REPLAY_PERIOD_MS 100       // ... the replay goes faster so a segment takes seconds

// TensorFlow Lite Micro working memory. The sketch prints how much the model really used
// (`arena used`); shrink this to that value plus a margin once measured on your board.
#define TENSOR_ARENA_BYTES 8192

// Run the recorded segments through the model at boot and compare with the scores the TFLite
// interpreter gave them on the host (AN_SCORE_TOL). Prints PASS/FAIL once, then starts the loop.
#define SELF_CHECK 1

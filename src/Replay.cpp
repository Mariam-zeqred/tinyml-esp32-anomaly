#include "Replay.h"

#include <Arduino.h>

#include "Config.h"
#include "test_vectors.h"

#if USE_REPLAY
static int g_seg = 0, g_pos = 0;

bool read_next(Reading& r) {
  const an_test_segment_t& seg = AN_TEST_SEGMENTS[g_seg];
  r.temp_c = seg.temp[g_pos];
  r.heater_on = seg.heater[g_pos] != 0;
  r.segment = seg.name;
  r.segment_start = g_pos == 0;
  if (++g_pos >= seg.n) {
    g_pos = 0;
    g_seg = (g_seg + 1) % AN_N_TEST_SEGMENTS;
  }
  return true;
}
#else
// Replace with your sensor: read the chamber temperature and the relay state the firmware
// commanded for this sample (the model needs both). Not run in this repo's tests.
bool read_next(Reading& r) {
  r.temp_c = 0.0f;
  r.heater_on = false;
  r.segment = "sensor";
  r.segment_start = false;
  return false;
}
#endif

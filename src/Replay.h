// Where readings come from. With USE_REPLAY the recorded segments are played back in a loop;
// otherwise `read_sensor` is the one function to replace with your sensor code.
#pragma once
#include <stdint.h>

struct Reading {
  float temp_c;
  bool heater_on;
  const char* segment;  // name of the replayed segment, or "sensor"
  bool segment_start;   // first reading of a segment: the window is reset
};

// Fills `r`; returns false when no reading is available yet (sensor not ready).
bool read_next(Reading& r);

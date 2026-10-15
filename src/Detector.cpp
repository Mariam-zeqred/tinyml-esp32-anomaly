#include "Detector.h"

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include "Window.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
tflite::MicroErrorReporter g_error_reporter;
tflite::MicroMutableOpResolver<2> g_resolver;  // the model is 4x FULLY_CONNECTED + 3x TANH
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
}  // namespace

bool Detector::begin(uint8_t* arena, size_t arena_bytes) {
  const tflite::Model* model = tflite::GetModel(g_anomaly_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("model schema %lu != supported %d\n", (unsigned long)model->version(),
                  TFLITE_SCHEMA_VERSION);
    return false;
  }
  g_resolver.AddFullyConnected();
  g_resolver.AddTanh();
  static tflite::MicroInterpreter interpreter(model, g_resolver, arena, arena_bytes,
                                              &g_error_reporter);
  g_interpreter = &interpreter;
  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed (raise TENSOR_ARENA_BYTES)");
    return false;
  }
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  if (g_input->type != kTfLiteInt8 || g_input->bytes != AN_N_FEATURES ||
      g_output->type != kTfLiteInt8 || g_output->bytes != AN_N_TARGETS) {
    Serial.println("model tensors do not match model_params.h");
    return false;
  }
  return true;
}

bool Detector::infer(const float f[AN_N_FEATURES], float recon[AN_N_TARGETS]) {
  if (!g_interpreter) return false;
  an_quantize(f, g_input->data.int8);
  const uint32_t t0 = micros();
  const bool ok = g_interpreter->Invoke() == kTfLiteOk;
  last_us_ = micros() - t0;
  if (ok) an_dequantize(g_output->data.int8, recon);
  return ok;
}

size_t Detector::arena_used() const { return g_interpreter ? g_interpreter->arena_used_bytes() : 0; }

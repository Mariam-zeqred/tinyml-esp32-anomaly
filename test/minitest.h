// Minimal assertion macros, so the host tests need no package download.
// (PlatformIO runs the same file under `test_framework = custom`.)
#pragma once
#include <stdio.h>
#include <string.h>

static int mt_failed = 0;
static int mt_passed = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (cond) {                                                                \
      mt_passed++;                                                             \
    } else {                                                                   \
      mt_failed++;                                                             \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                   \
    }                                                                          \
  } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

#define CHECK_STR_EQ(a, b)                                                     \
  do {                                                                         \
    if (strcmp((a), (b)) == 0) {                                               \
      mt_passed++;                                                             \
    } else {                                                                   \
      mt_failed++;                                                             \
      printf("FAIL %s:%d:\n    expected: %s\n    actual:   %s\n", __FILE__, __LINE__, (b), (a)); \
    }                                                                          \
  } while (0)

#define RUN(fn)                                                                \
  do {                                                                         \
    printf("  %s\n", #fn);                                                     \
    fn();                                                                      \
  } while (0)

static inline int mt_summary() {
  printf("%d checks passed, %d failed\n", mt_passed, mt_failed);
  return mt_failed == 0 ? 0 : 1;
}

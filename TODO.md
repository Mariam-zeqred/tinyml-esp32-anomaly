# TODO

Everything in the PLAN.md spec is in the repo (autoencoder on repo #1's real day, int8 TFLite
Micro export, ESP32 sketch, host-side replay test, footprint table, fault results, limits).
What is left needs the board or is housekeeping.

## Needs the device or Mariam

1. **Flash and measure.** `pio run -e esp32dev -t upload`, open the serial monitor, read the
   `arena used` and `inference ... us` lines the sketch prints at boot, and replace the two
   "estimate" rows of the README "Footprint on the ESP32" table with the measured values.
2. **Record a real fault.** The 2026-10-03 day contains no recorded fault; every physical fault
   is synthesised from repo #1's fitted thermal model. One real door opening or one real
   heater-off hour, exported from the device, would turn the fault table from model-based into
   measured for that fault kind.

## Housekeeping (QC 2026-10-15)

3. The firmware size in the README (334 009 B flash, 30 184 B static RAM) comes from the CI
   `pio run -e esp32dev` log and is stored nowhere in the repo. Paste the `RAM:` / `Flash:`
   lines of the next CI build into `figures/firmware_size.txt` so a reader can check it offline.
4. WALKTHROUGH.md says the first design (reconstructing the heater bits too) had "a baseline ten
   times lower"; that run is not in the repo. Either label it "(from that earlier run, not in
   the repo)" or drop the number. The file is at the 600-word cap, so trim elsewhere.
5. STYLE.md asks embedded repos for a `secrets.example.h`. The sketch needs no credentials
   (no Wi-Fi), which `.gitignore` says; add a one-line `include/secrets.example.h` saying so if
   the rule should be met literally.
6. Hand-written code is ~1 760 lines (generated C excluded), about three times the
   "small project" guideline. Not a defect, but worth saying in an interview.
7. `pio test -e native` cannot run in the cloud sandbox (PlatformIO registry blocked); CI runs
   it. Locally: `pip install platformio && pio test -e native`.

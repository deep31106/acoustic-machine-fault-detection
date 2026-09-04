# Acoustic Machine Fault Detection

Acoustic machine condition monitoring using digital signal processing and machine learning.

## Project Goal

Investigate whether acoustic signals can be used to detect abnormal machine operation and how robust such detection remains when recording and operating conditions change.

## Current Status

### Phase 0 — Dataset and Environment Setup

Completed initial inspection of the MIMII dataset using a small subset of:

- Machine: Fan
- Machine ID: id_00
- Condition: 0 dB
- Classes: Normal / Abnormal

Initial inspection confirms:

- Sampling rate: 16 kHz
- Recording duration: 10 seconds
- 160,000 samples per recording

The recordings have been loaded, visualized and inspected programmatically.

## Dataset

MIMII:
https://zenodo.org/records/3384388

Paper:
https://arxiv.org/abs/1909.09347

Dataset license:
CC BY-SA 4.0

The dataset is not included in this repository.

## Project Structure

```text
data/
    raw/
notebooks/
src/
results/
tests/
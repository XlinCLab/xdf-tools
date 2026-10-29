# xdf-tools

## Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
  * [Setup](#setup)
* [Modules](#modules)
* [Tests](#tests)

## Introduction
Tools for working with [Extensible Data Format (XDF)](https://github.com/sccn/xdf) recordings of time-synchronized
[LSL](https://labstreaminglayer.org/) streams (e.g. audio, video, EEG, eyetracking).

## Installation
Requires Python 3.10+. Optional extras:

| Extra | Adds | For |
|---|---|---|
| `audio` | `scipy` | WAV export, audio-stream utilities |
| `plot` | `pyqtgraph`, `PyQt6` | stream plotting |
| `test` | `pytest` | running tests |


### Setup

macOS / Linux:
```bash
./setup.sh
```

Windows (PowerShell)
```powershell
.\setup.ps1
```

The setup script creates a virtual environment in `.venv` and installs `xdf-tools` with all extras. To install only some extras, pass them as an argument
(e.g. `./setup.sh audio`, or `./setup.sh ""` for the core install only; on Windows `.\setup.ps1 -Extras audio`).

Then activate the environment:

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\Activate.ps1
```

## Modules

| Module | Contents |
|---|---|
| `constants` | `pyxdf` stream-dict keys and other string or numeric constants |
| `stream` | `XDFStream` |
| `file` | `XDFFile` |
| `clock` | drift-correction fit, raw/relative time → synchronized time conversion |
| `timing` | gap, jitter, and rate-mismatch validation and reports |
| `audio` | channel extraction, WAV export, `XDFAudioStream` |
| `indexing` | stream indexing utilities |
| `filters` | stream filtering utilities |
| `plotting` | pyqtgraph stream plots |


## Tests

With the environment from `setup.sh` activated (or after `pip install -e ".[test,audio]"`):

```bash
pytest
```

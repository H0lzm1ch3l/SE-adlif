# SE-adLIF

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

## Baronig, Ferrand, Sabathiel & Legenstein (2024):
**Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation**

---

## 📌 Getting Started

### Install Dependencies
To set up the required environment, run:
```sh
conda env create -f environment.yml
```

---

## 🔄 Reproducing Results

### SHD, SCC, ECG, BSD, and Oscillation Experiments
To start an experiment, use:
```sh
python run.py experiment=<experiment_name> ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir
```
**Notes:**
- `datadir` is **mandatory** and should contain the datasets.
- For SHD and SSC, data is downloaded automatically if not found at `datadir/SHDWrapper`.
- BSD dataset is created on the fly, so `datadir` can point to an empty directory.
- Results are stored in a local `results` folder unless `resultdir` is specified.
- `<experiment_name>` refers to configurations in `./config/experiment/`.

### Configuration Overrides (Hydra)
We use [Hydra](https://hydra.cc/) for configuration management. To override parameters, use the `++` syntax. For example, to change the number of training epochs:
```sh
python run.py experiment=SHD_SE_adLIF_small ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir ++n_epochs=10
```
For the BSD task with a different number of classes (Figure 6b):
```sh
python run.py experiment=BSD_SE_adLIF ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir ++dataset.num_classes=10
```

---

## 🎵 Audio Compression Experiments
To start an audio compression experiment, use:
```sh
python run_compress.py experiment=<experiment_name> ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir
```
### Available Configurations
- **SE-adLIF**: `compress_libri_SE_adLIF`
- **EF-adLIF**: `compress_libri_EF_adLIF`
- **LIF**: `compress_libri_LIF`

---

## 📊 Evaluating Audio Compression Models 
Model checkpoints for each configuration are available at [checkpoints](https://github.com/IGITUGraz/SE-adlif/tree/main/checkpoints).

### 1️⃣ Generating Wave Files
To generate wave files from a checkpoint:
```sh
generate_waves.py ckpt_path=/path/to/ckpt/example.ckpt source_wave_path=/path/to/libritts/location/ pred_wave_path=/path/to/prediction/ encoder_only=$encoder_flag
```
- `$encoder_flag`: `true` or `false`.
- `test_dir_path` can be a single `.wav` file or a directory containing `.wav` files.
- If no valid `.wav` files exist, the clean test-set from LibriTTS (~9h of audio) is used.

### 2️⃣ Evaluating Generated Waves
Use `evaluate_metrics.py` to compute SI-SNR or [Visqol](https://github.com/google/visqol):
```sh
evaluate_metrics.py metric=$metric source_wave_path=path/to/source/waves pred_wave_path=path/to/model/predictions
```
- `$metric` can be `si_snr` or `visqol`.
- **Note:** Visqol must be compiled manually following [these instructions](https://github.com/google/visqol). Additionally, the project requires either `gcc-9`/`g++-9` or `gcc-10`/`g++-10`. Set the compiler using:
```sh
export CC=gcc-9 CXX=g++-9
```
Furthermore, Visqol relies on Bazel but references an outdated HTTP resource (Armadillo) in its WORKSPACE file. 
The ressource has been moved [here](https://sourceforge.net/projects/arma/files/retired/armadillo-9.900.1.tar.xz.RETIRED/download).
You should modify the WORKSPACE file to reference your local copy as instructed [here](https://github.com/google/visqol/issues/117#issuecomment-2407779701).

---

## ⚠️ Important Information

### 🔹 Configuring `main.yaml`
Global parameters (e.g., `device: 'cpu'`, `cuda:0`) can be set in `config/main.yaml`. These settings are used by PyTorch Lightning’s `SingleDeviceStrategy`.

### 🔹 Block Index Padding
For variable-length sequences (e.g., SHD, SSC), a **custom masking procedure** is used:
- **Data vector:** Contains actual data, padded with zeros.
- **Block index (`block_idx`)**: Indicates valid data (1s) and padding (0s).
- **Target vector**: Maps indices to corresponding labels.

#### Example 1: Single Block Targeting
```
data vector: |1011010100101001010000000000000|
             |-----data---------|--padding---|
             ---> time

block_idx:   |1111111111111111111100000000000|
target: [-1, 3]
```
**Explanation:**
- Block `0` has target `-1` (ignored).
- Block `1` has target class `3`.

#### Example 2: Per-Timestep Labeling (ECG Task)
```
data vector: |1 0 1 1 0 0 1 0 0 0 0 0 0|
             |-----data---|--padding---|
             ---> time

block_idx:   |1 2 3 4 5 6 7 0 0 0 0 0 0|
target: [-1, 4, 3, 1, 3, 4, 6, 3]
```
**Explanation:**
- Multiple blocks (`1-7`) have corresponding target labels.
- Padding (`0s`) is ignored during loss computation.

Using this method, **per-block predictions** can be efficiently gathered using `torch.scatter_reduce`, ignoring padded time steps.

---



This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

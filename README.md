# UC Endoscopy Review — ResNet-18

Runnable ResNet-18 model for four-level Mayo endoscopic subscore classification of static ulcerative colitis images, with Grad-CAM, regional SHAP and an English Windows desktop application.

## Release scope

This release provides one trained model, the H configuration with seed 29. A/B/H model definitions and training configurations for seeds 17, 29 and 43 are included. Research images, original data partitions and the full paper evaluation pipeline remain outside this release. See `docs/OPEN_SOURCE_SCOPE.md`.

## Windows application

Download the English Windows ZIP from the repository's Releases page, extract it completely and open `UC_Image_Analysis.exe`. Keep `_internal` and `models` beside the executable. No separate Python or GPU installation is required.
Select your own static UC image, choose a writable export directory, and select **Predict + CAM** or **Predict + CAM + SHAP**. The application has no language or model-configuration selector. Results and `report_en.html` are saved locally.
No test images or screenshots are distributed.

## Run from source

Use 64-bit Python 3.12 with Tcl/Tk. Create a separate virtual environment and install dependencies:

```bash
python -m venv .venv
# Activate .venv using your operating system's standard activation command.
python -m pip install -r requirements.txt
python src/application/desktop.py
```

Download [resnet18_H_seed29.pt](https://github.com/qqbs-vop/uc-endoscopy-review-resnet18/releases/download/v1.0.0/resnet18_H_seed29.pt) from Releases and place it in `models/` before running the source application. Its SHA-256 is checked before inference. The Windows application ZIP already includes this model.
For local command-line prediction:

```bash
python scripts/predict.py --image /path/to/your/image.jpg --output /path/to/new/results
python scripts/predict.py --image /path/to/your/image.jpg --output /path/to/new/shap_results --shap
```

Each output directory must be new. SHAP takes longer than classification and CAM.

## Training on user-supplied data

Training uses Linux. Supply your own labeled image manifests using `docs/INPUT_SCHEMA.md`, update `configs/training.json`, and obtain the original ImageNet initialization:

```bash
python scripts/download_pretrained.py
python scripts/train.py --plan
python scripts/train.py --run resnet18_H_seed29 --device cpu
```

For GPU training, install a compatible GPU-enabled PyTorch 2.5.1 environment using `requirements-gpu.txt`. On a Slurm cluster, submit `scripts/train_slurm.sh` with the cluster's actual partition. The script preserves the scheduler's GPU visibility and uses the allocated logical device. No server address or credentials are embedded.

## Configuration

`configs/training.json` records AdamW learning rate 0.0001, batch size 32, weight decay 0.0001, a 500-epoch maximum, early-stopping patience 20, and auxiliary loss coefficient 0.3. The classifier and training initializer are shared across A/B/H within a seed.
`configs/explanation.json` records the explanation configuration. Technical metadata is separate from the desktop interface.

## Verification

```bash
python -m unittest discover -s tests -v
```

The publication audit records checks actually performed for this release. The locked Windows environment is recorded in `environment/windows-tested.lock.txt`; a fresh online installation and independent-machine compatibility testing are separate checks.

## Data and citation

Obtain source data from LIMUC at https://zenodo.org/records/5827695 or HyperKvasir at https://datasets.simula.no/hyper-kvasir/. Only the user-supplied image interface and manifest schema are distributed here.
Software citation metadata is provided in `CITATION.cff`. An article DOI will be added when available.

## License

Original project code and the released fine-tuned parameter artifact are provided under MIT. Third-party software, original pretrained initialization and dataset materials retain their respective terms. The Windows asset includes third-party license notices.

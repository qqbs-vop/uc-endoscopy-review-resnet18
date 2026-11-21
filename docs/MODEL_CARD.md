# Released model

Architecture: ResNet-18 with a four-class linear classifier and an auxiliary 512-to-128-to-1 regression branch.
Released configuration: H, seed 29. Fine-tuned weight SHA-256: `9897d7eb1a9038fabce42b22bd30cda2b9b89eec6722607fdeb48e5e61142d7f`.
Label order: MES 0, 1, 2, 3. The interface displays raw softmax probabilities and their MES 2 plus MES 3 sum.
Input processing resizes the long side to 224 pixels, preserves aspect ratio, applies mean-color letterboxing, and uses ImageNet normalization.
Training uses cross-entropy plus 0.3 times Huber loss on numeric MES labels, with delta 1.0.
The A and B training configurations are provided as code/configuration; their trained weights are outside this release.
CAM supports the predicted class and the MES-at-least-2 probability target. SHAP uses an 8-by-8 spatial grid with a Gaussian-blurred baseline and 64 permutations by default.
SHAP regions are spatial image regions rather than named clinical features.
Intended use: research and physician review of static UC endoscopy images.
Release verification covers numerical agreement, classification, CAM, regional SHAP and application startup. Study performance evaluation is documented separately in the associated manuscript.

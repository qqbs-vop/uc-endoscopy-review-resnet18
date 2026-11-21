# User-supplied training inputs

Configure your own image root and patient-separated train/validation manifests in `configs/training.json`.
CSV fields: `image_id`, `relative_path`, `patient_id`, `mes_grade`, `boundary_label`, `eligible_ordinal`, `sha256`.
Image identifiers must be unique within each manifest. Patient IDs, image IDs and nonempty image hashes must not overlap between the two partitions.
MES is an integer from 0 to 3. The boundary label is 1 for MES 2/3 and 0 for MES 0/1. Training/validation rows require `eligible_ordinal=1`.
The optional manifest `rows` and `sha256` fields enable count/hash verification. Set them to your own manifest values.
No original patient split, image identifier list, patient-level records or images are included in this release.

Public dataset sources: LIMUC, https://zenodo.org/records/5827695 and HyperKvasir, https://datasets.simula.no/hyper-kvasir/.
Follow their source terms and cite their associated dataset publications when using those datasets.

# LAYER identity-probe dataset

The committed default dataset is:

```text
identity_layers/datasets/wikimedia_identity_probe/
```

Current contents:

- 15 identities
- 3 images per identity
- 45 total images
- 45 same-identity pairs
- 945 different-identity pairs
- all processed images are 512×512 JPEGs

Main files:

```text
identity_layers/datasets/wikimedia_identity_probe/identity_manifest.csv
identity_layers/datasets/wikimedia_identity_probe/same_identity_pairs.csv
identity_layers/datasets/wikimedia_identity_probe/different_identity_pairs.csv
identity_layers/datasets/wikimedia_identity_probe/dataset_summary.json
identity_layers/datasets/wikimedia_identity_probe/source_attribution.csv
identity_layers/datasets/wikimedia_identity_probe/dataset_contact_sheet.jpg
```

The images were selected from Wikimedia Commons using the MediaWiki API. Per-file license, author/artist, credit, source URL, and Commons description URL are recorded in:

```text
identity_layers/datasets/wikimedia_identity_probe/source_attribution.csv
```

The scan configs point to this manifest by default:

```text
identity_layers/configs/layer_inventory.json
identity_layers/configs/identity_scan.json
```

You can rebuild the dataset, if needed, with:

```bash
python -m layer.scripts.build_wikimedia_identity_dataset \
  --root /home/interns/Desktop/layer \
  --identities 15 \
  --images-per-identity 3
```

The preprocessing uses generic resizing/letterbox padding only. It does not use face detection, face alignment, landmarks, or paid image sources.

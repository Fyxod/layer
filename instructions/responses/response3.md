# Response 3 — dataset built before A0/A1

Built and committed the default identity-probe dataset for LAYER.

Dataset:

```text
identity_layers/datasets/wikimedia_identity_probe/
```

Contents:

- 15 identities
- 3 images per identity
- 45 total processed images
- 45 same-identity pairs
- 945 different-identity pairs
- 512×512 JPEG inputs
- `source_attribution.csv` with Commons title, URL, license, artist/credit fields
- `dataset_contact_sheet.jpg` for quick visual inspection

The configs now point to:

```text
identity_layers/datasets/wikimedia_identity_probe/identity_manifest.csv
```

So the normal A0/A1 and Milestone 1 commands will use this dataset automatically after `git pull`.

No full model scan was run on Windows.

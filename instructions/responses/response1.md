# Response 1 — Milestone 1 implementation note

Implemented the first LAYER milestone as an InstructPix2Pix identity-layer scan.

What this includes:

- InstructPix2Pix layer inventory with frozen model checks.
- Forward-hook activation shape inventory.
- Pooled activation extraction for selected identity pairs.
- Same-identity vs different-identity scoring.
- Initial layer/timestep ranking and HTML/Markdown report generation.

What this intentionally does not include:

- No geometry attack.
- No pixel noise, patches, finetuning, LoRA, SPSA, CEM, landmarks, face alignment, or face detection.
- No model-weight training.

The default MAT auto-manifest uses available image variants for `face_002` and `face_005` so the code path can produce same-identity development pairs. Treat this as a diagnostic scan unless a richer identity dataset is supplied.

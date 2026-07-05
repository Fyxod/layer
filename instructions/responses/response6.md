# Response 6 — Stage B target decision

Stage A is complete with ArcFace baseline support.

The first Stage-B implementation should target only:

```text
vae_image_latent
unet.conv_in
```

Rationale:

- `vae_image_latent` and `unet.conv_in` are the only layers that combine identity separation with nonzero useful geometry gradients in A7.
- Some deeper layers correlate more strongly with ArcFace pairwise distances, especially `unet.mid_block` and `unet.up_blocks.0`, but they failed the current gradient gate.

Decision to remember:

```text
Stage B should start with vae_image_latent and unet.conv_in only.
If those fail, revisit the ArcFace-correlated deeper layers with a modified gradient probe.
```

The next instruction file is:

```text
instructions/a6000_stage_b_next_commands.md
```

Those Stage-B A6000 commands are placeholders for after the Stage-B scripts are implemented. Do not run them yet.

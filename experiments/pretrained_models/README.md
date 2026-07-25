# Pretrained models

- `spynet_sintel_final.pth` is the BasicSR-listed SPyNet checkpoint. Expected
  SHA-256:
  `3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714`.
- Put the released generator checkpoint at
  `frequency_guided_restoration.pth`, or update `pretrain_network_g` in each
  test YAML.

Large generator checkpoints should normally be published as a GitHub Release
asset rather than committed to Git.

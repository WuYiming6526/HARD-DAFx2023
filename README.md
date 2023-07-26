# Self-Supervised Disentanglement of Harmonic and Rhythmic Features in Music Audio Signals

Source codes for the DAFx2023 conference paper "Self-Supervised Disentanglement of Harmonic and Rhythmic Features in Music Audio Signals".

# How to train

1. Store the audio files uncer a certain path,
2. In `configs/disentangle-2dvaegan-mono.yaml`, modify `model.init_args.dset_config.path` to specify the audio path,
3. Run the following command.

```
python RunCLI.py fit --config configs/disentangle-2dvaegan-mono.yaml
```

# Inference

An example code of audio generation is shown in `Inference.ipynb`.
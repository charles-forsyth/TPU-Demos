# TPU Demos

Research Computing AI/ML demos using Google TPUs (v5e).

## Environment Setup

These demos are optimized for Google Cloud TPU v5e environments.

```bash
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax optax
```

## Demos

1. **Biology / Medical Imaging**: ResNet-50 implementation on JAX/Flax for high-throughput image classification.

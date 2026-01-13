import jax
import jax.numpy as jnp

from tpu_demos.biology.medical_imaging import ResNet50


def test_resnet_output_shape():
    model = ResNet50(num_classes=10)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 224, 224, 3))
    variables = model.init(rng, x, train=False)
    out = model.apply(variables, x, train=False)
    assert out.shape == (2, 10)

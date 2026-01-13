import jax
import jax.numpy as jnp

from tpu_demos.biology.medical_imaging import ResNet50, training_loop


def test_resnet_output_shape():
    model = ResNet50(num_classes=10)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 224, 224, 3))
    variables = model.init(rng, x, train=False)
    out = model.apply(variables, x, train=False)
    assert out.shape == (2, 10)


def test_training_loop_yields():
    """Ensure the training loop yields correct metrics structure."""
    iterator = training_loop(num_steps=2, batch_size=2)

    # First yield is compilation status
    first = next(iterator)
    assert first["status"] == "COMPILING"
    assert first["step"] == 0

    # Second yield is first step (after compile)
    second = next(iterator)
    assert second["status"] == "RUNNING"
    assert second["step"] == 1
    # Note: compile_time was removed for simplicity in polling refactor

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


class ResNetBlock(nn.Module):
    """Simple ResNet Block."""

    filters: int
    conv: Callable[..., Any]
    norm: Callable[..., Any]
    act: Callable[[jnp.ndarray], jnp.ndarray]
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm(use_running_average=not train)(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(use_running_average=not train)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides)(residual)
            residual = self.norm(use_running_average=not train)(residual)

        return self.act(residual + y)


class ResNet50(nn.Module):
    """Simplified ResNet-50 for TPU demonstration."""

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        conv = nn.Conv
        norm = nn.BatchNorm

        x = conv(64, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
        x = norm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")  # type: ignore[no-untyped-call]

        # Simplified stages for demonstration
        for filters, blocks in zip([64, 128, 256, 512], [3, 4, 6, 3]):
            for i in range(blocks):
                strides = (2, 2) if i == 0 and filters != 64 else (1, 1)
                x = ResNetBlock(
                    filters=filters, conv=conv, norm=norm, act=nn.relu, strides=strides
                )(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


def create_train_state(
    rng: jax.Array, learning_rate: float, momentum: float
) -> tuple[Any, optax.GradientTransformation, nn.Module]:
    """Initializes training state."""
    model = ResNet50()
    variables = model.init(rng, jnp.ones([1, 224, 224, 3]), train=False)
    params = variables["params"]
    tx = optax.sgd(learning_rate, momentum)
    return params, tx, model


@jax.jit
def train_step(
    params: Any,
    opt_state: optax.OptState,
    batch: Mapping[str, jnp.ndarray],
    tx: optax.GradientTransformation,
    model: nn.Module,
) -> tuple[Any, optax.OptState, jnp.ndarray]:
    """A single training step compiled with XLA."""

    def loss_fn(params: Any) -> jnp.ndarray:
        logits = model.apply({"params": params}, batch["image"], train=True)
        loss: jnp.ndarray = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def main() -> None:
    print("TPU Medical Imaging Demo: ResNet-50")
    print(f"TPU Devices: {jax.devices()}")

    # Initialize state
    rng = jax.random.PRNGKey(0)
    params, tx, model = create_train_state(rng, 0.1, 0.9)
    opt_state = tx.init(params)

    # Dummy data
    batch = {
        "image": jnp.ones((8, 224, 224, 3), dtype=jnp.bfloat16),
        "label": jnp.zeros((8,), dtype=jnp.int32),
    }

    # Train
    params, opt_state, loss = train_step(params, opt_state, batch, tx, model)
    print(f"Initial loss: {loss}")


if __name__ == "__main__":
    main()

import json
import time
from functools import partial
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from rich.console import Console

console = Console()


class ResNetBlock(nn.Module):
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
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        conv = nn.Conv
        norm = nn.BatchNorm

        x = conv(64, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
        x = norm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")  # type: ignore

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
) -> tuple[Any, Any, optax.GradientTransformation, nn.Module]:
    model = ResNet50()
    variables = model.init(rng, jnp.ones([1, 224, 224, 3]), train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    tx = optax.sgd(learning_rate, momentum)
    return params, batch_stats, tx, model


@partial(jax.jit, static_argnames=["tx", "model"])
def train_step(
    params: Any,
    batch_stats: Any,
    opt_state: optax.OptState,
    batch: Mapping[str, jnp.ndarray],
    tx: optax.GradientTransformation,
    model: nn.Module,
) -> tuple[Any, Any, optax.OptState, jnp.ndarray]:
    def loss_fn(params: Any, batch_stats: Any) -> tuple[jnp.ndarray, Any]:
        logits, new_model_state = model.apply(
            {"params": params, "batch_stats": batch_stats},
            batch["image"],
            train=True,
            mutable=["batch_stats"],
        )
        loss: jnp.ndarray = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, new_model_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_model_state), grads = grad_fn(params, batch_stats)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, new_model_state["batch_stats"], opt_state, loss


def main() -> None:
    console.print("[bold cyan]Initializing TPU Job...[/]")
    try:
        console.log(f"JAX Devices: {jax.devices()}")
    except Exception:
        console.log("[yellow]Could not detect TPU devices. Using CPU fallback?[/]")

    rng = jax.random.PRNGKey(0)
    params, batch_stats, tx, model = create_train_state(rng, 0.1, 0.9)
    opt_state = tx.init(params)

    batch_size = 128
    batch = {
        "image": jnp.ones((batch_size, 224, 224, 3), dtype=jnp.bfloat16),
        "label": jnp.zeros((batch_size,), dtype=jnp.int32),
    }

    console.log("Compiling training step...")
    start_compile = time.time()
    params, batch_stats, opt_state, loss = train_step(
        params, batch_stats, opt_state, batch, tx, model
    )
    jax.block_until_ready(loss)  # type: ignore
    console.log(f"Compilation finished in {time.time() - start_compile:.2f}s")

    console.log("Starting training loop (300 steps)...")
    start_time = time.time()

    final_loss = 0.0
    for step in range(1, 301):
        params, batch_stats, opt_state, loss = train_step(
            params, batch_stats, opt_state, batch, tx, model
        )
        # Log every 50 steps
        if step % 50 == 0:
            jax.block_until_ready(loss)  # type: ignore
            console.log(f"Step {step}: Loss = {loss:.4f}")
            final_loss = float(loss)

    total_time = time.time() - start_time
    avg_throughput = (300 * batch_size) / total_time

    console.print(
        f"[bold green]Job Complete![/] Average Throughput: {avg_throughput:.2f} img/s"
    )

    # Save results
    results = {
        "final_loss": final_loss,
        "avg_throughput": avg_throughput,
        "total_time": total_time,
        "status": "SUCCESS",
    }
    with open("results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

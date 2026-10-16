# Quantizations

Quantizations provides weight-storage codecs for PLeX applications. Each format
is an independent package, so applications depend only on the formats they use.

## Available formats

| Package | Format | Typical use |
|---|---|---|
| `F32` | 32-bit floating point | Reference and compatibility builds. |
| `F16` | IEEE half precision | Reduced storage and compute on supported devices. |
| `BF16` | Brain floating point | Checkpoints distributed in BF16. |
| `Int8` | Symmetric 8-bit integer | Per-output-row compressed weights. |
| `Q4K` | Affine 4-bit groups | Aggressive compression for compatible matrices. |
| `Q8` | Grouped 8-bit | Lower-risk compressed projection weights. |
| `Q5K` | GGML Q5_K | Five-bit K-block checkpoints. |
| `Q6K` | GGML Q6_K | Six-bit K-block checkpoints. |
| `KBlock` | Q5_K/Q6_K facade | Select K-block width through one dependency. |

## Add formats

```toml
[dependencies]
quantizations = { git = "https://github.com/useplx/Quantizations.git", version = "v1.0.0" }
```

## Read a checkpoint

List formats required by the source checkpoint:

```toml
[myelin]
formats = ["bf16", "f32"]
```

This does not choose final artifact storage. It only tells the compiler which
source formats may appear.

## Select artifact storage

```toml
[[policies."model.layers.*.mlp.down_proj.weight".schemes]]
scope = "constant"
lifetime = "immutable"
update = "none"
format = "q4k"
```

The `format` value names an export provided by one of the resolved packages.

## Choose a format

Use this validation sequence:

1. build and test the model in its original checkpoint format;
2. identify the largest compatible projection weights;
3. apply a compressed format to those paths;
4. keep small normalization and bias parameters at higher precision;
5. compare output quality, artifact size, memory, and latency;
6. store each deployment profile in its own manifest.

A smaller format does not guarantee faster execution. Results depend on device
support, model shapes, and the generated graph.

## Important constraints

- A format can reject incompatible tensor shapes.
- K-block formats require weights that fit their block layout.
- Tied weights share one stored representation.
- Codecs for immutable weights do not automatically apply to mutable model
  state.
- Quantization changes model output and must be evaluated for the target task.

## Create a codec

A codec package publishes one named format from its `synps.toml`:

```toml
[build.graph.format]
toolchain = "cpp"
sources = ["src/package.cpp"]
exports = ["my-format"]

[capabilities]
provides = ["quantization"]
```

See [Create a package](https://useplx.github.io/guide/create-a-package/) and the
[quantization guide](https://useplx.github.io/guide/quantize-weights/).

## Documentation

Full documentation lives at **[https://useplx.github.io/packages/quantizations/](https://useplx.github.io/packages/quantizations/)**.

- [Format overview](https://useplx.github.io/packages/quantizations/)
- [Quantization guide](https://useplx.github.io/guide/quantize-weights/)
- [Create a package](https://useplx.github.io/guide/create-a-package/)

## License

Quantizations is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial use is permitted
without charge. Commercial use requires a separate agreement with the copyright
holder.

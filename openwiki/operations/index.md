# Files

- [Dependencies, extras and version compatibility](dependencies-and-compatibility.md) - What each runtime dependency and optional extra is for, which upstream dependencies were replaced and how, the code seams that keep the library working on transformers 4.x and 5.x, flash-attn wheel constraints, and the stacks that were tested.
- [Model directory, loading and caches](model-loading-and-caches.md) - The files the library reads from model_dir and config.yaml, how auxiliary models are resolved or downloaded, the order and rules of model loading (on-device construction, mmap, strict loading, weight-norm folding, bf16), and every in-memory and on-disk cache.

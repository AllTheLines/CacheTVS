# Viewshed Reconstruction

This crate is for reconstucting viewsheds from their raw "polar segments".

It is used for both in the Rust backend for outputting final viewsheds. And it is used in browser
frontends for rendering viewsheds in maps.

How to compile for WASM:
```
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

wasm-pack build \
  --target web \
  --out-dir /publicish/Workspace/viewview/website/src/lib/viewshed-reconstructor \
  crates/viewshed-reconstructor
```


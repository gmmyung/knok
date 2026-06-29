knok::generated_graphs!(pub mod graphs);
knok::generated_graphs!(pub mod mlir_models, "knok_mlir_models.rs");

#[cfg(test)]
mod common;
#[cfg(test)]
mod tests;

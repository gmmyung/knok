knok::generated_graphs!(pub mod graphs);
knok::generated_graphs!(pub mod autograd_graphs, "knok_autograd_graphs.rs");
knok::generated_graphs!(pub mod mlir_models, "knok_mlir_models.rs");
knok::generated_graphs!(pub mod tuned, "knok_tuned_graphs.rs");
knok::generated_graphs!(pub mod tuned_grad, "knok_tuned_grad_graphs.rs");

#[cfg(test)]
mod common;
#[cfg(test)]
mod tests;

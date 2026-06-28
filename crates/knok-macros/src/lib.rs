use proc_macro::TokenStream;

#[proc_macro]
pub fn mlir_model(input: TokenStream) -> TokenStream {
    knok_compile::expand_mlir_model(input.into()).into()
}

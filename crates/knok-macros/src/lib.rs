use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn graph(attr: TokenStream, item: TokenStream) -> TokenStream {
    knok_compile::expand_graph(attr.into(), item.into()).into()
}

#[proc_macro]
pub fn mlir_model(input: TokenStream) -> TokenStream {
    knok_compile::expand_mlir_model(input.into()).into()
}

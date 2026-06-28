knok::generated_graphs!(pub mod graphs);

#[cfg(test)]
mod tests {
    use knok::tensor::Tensor2;

    use super::graphs;

    #[test]
    fn generated_build_traced_graph_runs() {
        let x = Tensor2::from_array([[1.0, 2.0], [3.0, 4.0]]);
        let output = graphs::forward::call(x).unwrap();
        assert_eq!(output.as_slice(), &[8.0, 11.0, 16.0, 23.0]);
    }
}

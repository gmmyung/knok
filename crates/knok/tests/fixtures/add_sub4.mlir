module @imported {
  func.func @add_sub4(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %sum = arith.addf %arg0, %arg1 : tensor<4xf32>
    %diff = arith.subf %arg0, %arg1 : tensor<4xf32>
    return %sum, %diff : tensor<4xf32>, tensor<4xf32>
  }
}

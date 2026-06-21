module @imported {
  func.func @add4(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %0 = arith.addi %arg0, %arg1 : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

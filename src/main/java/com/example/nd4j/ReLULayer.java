package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ReLULayer implements Layer {
  private INDArray out;

  @Override
  public INDArray forward(INDArray in) {
    this.out = Transforms.relu(in);
    return out;
  }

  @Override
  public INDArray backward(INDArray dout) {
    INDArray result = Nd4j.create(dout.shape());
    for(int i = 0; i < out.rows(); i++) {
      for(int j = 0; j < out.columns(); j++) {
        result.put(i, j, out.getDouble(i, j) <= 0 ? 0 : dout.getDouble(i, j));
      }
    }
    return result;
  }
}

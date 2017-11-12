package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SigmoidLayer implements Layer {
  private INDArray out;

  @Override
  public INDArray forward(INDArray in) {
    this.out = Transforms.sigmoid(in);
    return out;
  }

  @Override
  public INDArray backward(INDArray dout) {
    INDArray one = Nd4j.ones(out.shape());
    INDArray dx = dout.mul(one.sub(out)).mul(out);
    return dx;
  }
}

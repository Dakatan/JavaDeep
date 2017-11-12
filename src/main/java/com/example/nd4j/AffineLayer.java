package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public class AffineLayer implements Layer {
  private INDArray weight;
  private INDArray bias;
  private INDArray x;
  private INDArray differentialWeight;
  private double differentialBias;

  public AffineLayer(INDArray weight, INDArray bias) {
    this.weight = weight;
    this.bias = bias;
  }

  @Override
  public INDArray forward(INDArray x) {
    this.x = x;
    return x.mmul(weight).addRowVector(bias);
  }

  @Override
  public INDArray backward(INDArray dout) {
    INDArray dx = dout.mmul(weight.transpose());
    differentialWeight = x.transpose().mmul(dout);
    differentialBias = dout.sumNumber().doubleValue();
    weight = weight.sub(differentialWeight.mul(5));
    bias = bias.sub(differentialBias * 5);
    return dx;
  }
}

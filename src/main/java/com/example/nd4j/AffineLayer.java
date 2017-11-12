package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AffineLayer implements Layer {
  private INDArray weight;
  private INDArray bias;
  private INDArray x;
  private INDArray differentialWeight;
  private double differentialBias;
  private double learningRate = 0.1;

  public AffineLayer(int row, int col) {
    this.weight = Nd4j.randn(row, col);
    this.bias = Nd4j.randn(1, col);
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  public AffineLayer(int row, int col, double learningRate) {
    this.weight = Nd4j.randn(row, col);
    this.bias = Nd4j.randn(1, col);
    this.learningRate = learningRate;
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
    weight = weight.sub(differentialWeight.mul(learningRate));
    bias = bias.sub(differentialBias * learningRate);
    return dx;
  }
}

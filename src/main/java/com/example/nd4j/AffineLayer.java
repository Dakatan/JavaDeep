package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AffineLayer implements Layer {
  private INDArray weight;
  private INDArray bias;
  private INDArray in;
  private INDArray dWeight;
  private double dBias;
  private double learningRate = 0.1;
  private int row;
  private int col;

  public AffineLayer(int row, int col) {
    this.row = row;
    this.col = col;
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
  public INDArray forward(INDArray in) {
    this.in = in;
    return in.mmul(weight).addRowVector(bias);
  }

  @Override
  public INDArray backward(INDArray dout) {
    INDArray dx = dout.mmul(weight.transpose());
    dWeight = in.transpose().mmul(dout);
    dBias = dout.sumNumber().doubleValue();
    weight = weight.sub(dWeight.mul(learningRate));
    bias = bias.sub(dBias * learningRate);
    return dx;
  }
}

package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public class SquareErrorOutputLayer implements OutputLayer {
  private INDArray teacher;
  private INDArray out;
  private double loss;

  @Override
  public void setTeacher(INDArray teacher) {
    this.teacher = teacher;
  }

  @Override
  public double getError(INDArray x) {
    this.out = x;
    loss = out.sub(teacher).norm2Number().doubleValue();
    return loss;
  }

  @Override
  public INDArray getDout(double dout) {
    int size = teacher.rows();
    INDArray dx = out.sub(teacher).div(size);
    return dx;
  }
}
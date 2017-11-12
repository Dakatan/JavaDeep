package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FinalLayer {
  private INDArray teacher;
  private INDArray out;
  private double loss;

  public void setTeacher(INDArray teacher) {
    this.teacher = teacher;
  }

  public double forward(INDArray x) {
    this.out = x;
    loss = out.sub(teacher).norm2Number().doubleValue();
    return loss;
  }

  public INDArray backward(double dout) {
    int size = teacher.rows();
    INDArray dx = out.sub(teacher).div(size);
    return dx;
  }
}

package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SquareErrorOutputLayer implements OutputLayer {
  private INDArray teacher;
  private INDArray out;
  private double loss;

  @Override
  public void setTeacher(INDArray teacher) {
    this.teacher = teacher;
  }

  @Override
  public double calculateError(INDArray x) {
    this.out = Transforms.softmax(x);
    loss = out.sub(teacher).norm2Number().doubleValue();
    return loss;
  }

  @Override
  public INDArray calculateDout() {
    int size = teacher.rows();
    INDArray dx = out.sub(teacher).div(size);
    return dx;
  }
}

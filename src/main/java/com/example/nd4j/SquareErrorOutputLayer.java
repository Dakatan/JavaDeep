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
  public INDArray forward(INDArray in) {
    this.out = Transforms.softmax(in);
    loss = out.squaredDistance(teacher) / out.rows();
    return out;
  }

  @Override
  public INDArray backward(INDArray dout) {
    return backward();
  }

  @Override
  public INDArray backward() {
    int size = teacher.rows();
    INDArray dx = out.sub(teacher).div(size);
    return dx;
  }

  @Override
  public double getError() {
    return loss;
  }
}

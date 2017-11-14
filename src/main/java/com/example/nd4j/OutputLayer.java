package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface OutputLayer extends Layer {
  void setTeacher(INDArray teacher);
  double getError();
  INDArray backward();
}

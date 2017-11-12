package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface OutputLayer {
  void setTeacher(INDArray teacher);
  double calculateError(INDArray x);
  INDArray calculateDout();
}

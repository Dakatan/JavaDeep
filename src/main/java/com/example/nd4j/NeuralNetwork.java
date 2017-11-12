package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class NeuralNetwork<T extends ActivationFunctionLayer> {
  private OutputLayer outputLayer;
  private List<Layer> network;
  private Class<T> activationFuncType;
  private TrainingData trainingData;
  private int learningCount;

  public NeuralNetwork(int layerCount) {
  }

  public void learn() {
    for(int i = 1; i <= learningCount; i++) {
      outputLayer.setTeacher(trainingData.getTeacherData());
      outputLayer.calculateError(forward(trainingData.getInputData()));
      backward(outputLayer.calculateDout());
    }
  }

  private INDArray forward(INDArray in) {
    for(Layer layer : network) {
      in = layer.forward(in);
    }
    return in;
  }

  private INDArray backward(INDArray dout) {
    List<Layer> reverseNetwork = new LinkedList<>(network);
    Collections.reverse(reverseNetwork);
    for(Layer layer : reverseNetwork) {
      dout = layer.backward(dout);
    }
    return dout;
  }
}

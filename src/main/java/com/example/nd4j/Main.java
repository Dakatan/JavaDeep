package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class Main {
  public static void main(String[] args) {
    INDArray trainings = Nd4j.create(new double[][]{{1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}});
    INDArray teachers = Nd4j.create(new double[][]{{0.0}, {1.0}, {1.0}, {0.0}});
    double learningRate = 5;

    List<Layer> network = new LinkedList<>();
    network.add(new AffineLayer(2, 3, learningRate));
    network.add(new SigmoidLayer());
    network.add(new AffineLayer(3, 3, learningRate));
    network.add(new SigmoidLayer());
    network.add(new AffineLayer(3, 1, learningRate));
    network.add(new SigmoidLayer());

    OutputLayer finalLayer = new SquareErrorOutputLayer();
    finalLayer.setTeacher(teachers);

    for(int i = 1; i <= 5000; i++) {
      INDArray y = forward(network, trainings);
      finalLayer.setTeacher(teachers);

      double z = finalLayer.getError(y);
      INDArray dout = finalLayer.getDout(1.0);
      backward(network, dout);

      System.out.println(z);
    }
  }

  public static INDArray forward(List<Layer> network, INDArray x) {
    for(Layer layer : network) {
      x = layer.forward(x);
    }
    return x;
  }

  public static INDArray backward(List<Layer> network, INDArray dout) {
    List<Layer> reverseNetwork = new LinkedList<>(network);
    Collections.reverse(reverseNetwork);
    for(Layer layer : reverseNetwork) {
      dout = layer.backward(dout);
    }
    return dout;
  }
}

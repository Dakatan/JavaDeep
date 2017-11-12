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
    double q = 5;

    INDArray w1 = Nd4j.randn(2, 3);
    INDArray b1 = Nd4j.randn(1, 3);

    INDArray w2 = Nd4j.randn(3, 3);
    INDArray b2 = Nd4j.randn(1, 3);

    INDArray w3 = Nd4j.randn(3, 1);
    INDArray b3 = Nd4j.randn(1, 1);

    Layer affineLayer1 = new AffineLayer(w1, b1);
    Layer sigmoidLayer1 = new SigmoidLayer();

    Layer affineLayer2 = new AffineLayer(w2, b2);
    Layer sigmoidLayer2 = new SigmoidLayer();

    Layer affineLayer3 = new AffineLayer(w3, b3);
    Layer sigmoidLayer3 = new SigmoidLayer();

    List<Layer> network = new LinkedList<>();
    network.add(affineLayer1);
    network.add(sigmoidLayer1);
    network.add(affineLayer2);
    network.add(sigmoidLayer2);
    network.add(affineLayer3);
    network.add(sigmoidLayer3);

    FinalLayer finalLayer = new FinalLayer();
    finalLayer.setTeacher(teachers);

    for(int i = 1; i <= 1000; i++) {
      INDArray y = forward(network, trainings);
      double z = finalLayer.forward(y);
      INDArray dout = finalLayer.backward(1.0);
      backward(network, dout);

      System.out.println("COUNT: " + i);
      System.out.println(y);
      System.out.println(z);
      System.out.println();
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

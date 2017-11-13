package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class Main {
  public static void main(String[] args) throws Exception {
//    INDArray trainings = Nd4j.create(new double[][]{{1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}});
//    INDArray teachers = Nd4j.create(new double[][]{{0.0}, {1.0}, {1.0}, {0.0}});
    double learningRate = 0.1;

    List<Layer> network = new LinkedList<>();
    int IN = 784;
    int N1 = 100;
    int N2 = 100;
    int OUT = 10;
    network.add(new AffineLayer(IN, N1, learningRate));
    network.add(new SigmoidLayer());
    network.add(new AffineLayer(N1, N2, learningRate));
    network.add(new SigmoidLayer());
    network.add(new AffineLayer(N2, OUT, learningRate));
    network.add(new SigmoidLayer());

    OutputLayer outputLayer = new SquareErrorOutputLayer();

    MNIST mnist = new MNIST(MNIST.DataType.TRAIN);

    for(int i = 1; i <= 20000; i++) {
      int[] batchIndexes = createBatchIndex(100, 60000);

      INDArray y = forward(network, mnist.getFeatures(batchIndexes));
      outputLayer.setTeacher(mnist.getLabels(batchIndexes));

      double z = outputLayer.calculateError(y);
      INDArray dout = outputLayer.calculateDout();
      backward(network, dout);

      System.out.println(z);
    }
  }

  public static int[] createBatchIndex(int size, int max) {
    int[] batchIndex = new int[size];
    Random random = new Random();
    for(int i = 0; i < batchIndex.length; i++) {
      batchIndex[i] = random.nextInt(max);
    }
    return batchIndex;
  }

  public static INDArray forward(List<Layer> network, INDArray in) {
    for(Layer layer : network) {
      in = layer.forward(in);
    }
    return in;
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

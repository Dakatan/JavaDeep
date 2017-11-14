package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class Main {
  public static void main(String[] args) throws Exception {
    double learningRate = 0.1;

    List<Layer> network = new LinkedList<>();
    int IN = 784;
    int N1 = 100;
    int N2 = 100;
    int OUT = 10;
    OutputLayer outputLayer = new SquareErrorOutputLayer();
    network.add(new AffineLayer(IN, N1, learningRate));
    network.add(new SigmoidLayer());
    network.add(new AffineLayer(N1, N2, learningRate));
    network.add(new SigmoidLayer());
    network.add(new AffineLayer(N2, OUT, learningRate));
    network.add(outputLayer);

    MNIST mnist1 = new MNIST(MNIST.DataType.TRAIN);

    for(int i = 1; i <= 5000; i++) {
      int[] batchIndexes = createBatchIndex(100, 60000);
      outputLayer.setTeacher(mnist1.getLabels(batchIndexes));
      forward(network, mnist1.getFeatures(batchIndexes));
      System.out.println("COUNT: " + i + ", LOSS: " + outputLayer.getError());
      backward(network, null);
    }

    MNIST mnist2 = new MNIST(MNIST.DataType.TEST);
    for(int i = 0; i < 10; i++) {
      INDArray ans = forward(network, mnist2.getFeatures(i));
      System.out.println(ans);
      System.out.println(mnist2.getLabels(i));
      System.out.println();
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

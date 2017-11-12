package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.Random;

public class Practice {
  public static void main(String[] args) {
    for(int i = 0; i < 1; i++) {
      calc();
    }
  }

  public static void calc() {
    INDArray trainings = Nd4j.create(new double[][]{{1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}});
    INDArray teachers = Nd4j.create(new double[][]{{0.0}, {1.0}, {1.0}, {0.0}});
    double q = 5;

    INDArray w1 = Nd4j.randn(2, 3);
    INDArray b1 = Nd4j.randn(1, 3);

    INDArray w2 = Nd4j.randn(3, 3);
    INDArray b2 = Nd4j.randn(1, 3);

    INDArray w3 = Nd4j.randn(3, 1);
    INDArray b3 = Nd4j.randn(1, 1);

    Random random = new Random();

    for(int i = 1; i <= 1000; i++) {
      int index = random.nextInt(4);

      INDArray in = trainings.getRow(index);
      INDArray teacher = teachers.getRow(index);

      INDArray y1 = Transforms.sigmoid(in.mmul(w1).addRowVector(b1));
      INDArray y2 = Transforms.sigmoid(y1.mmul(w2).addRowVector(b2));
      INDArray y3 = Transforms.sigmoid(y2.mmul(w3).addRowVector(b3));

      if(i == 1 || i % 100 == 0) {
        System.out.println("COUNT: " + i);
        plot(trainings, teachers, w1, b1, w2, b2, w3, b3);
      }

      INDArray oneVector1 = Nd4j.ones(1, 1);
      INDArray delta3 = oneVector1.sub(y3).mul(y3).mul(y3.sub(teacher));
      b3 = b3.sub(delta3.mul(q));
      w3 = w3.sub(y2.transpose().mmul(delta3.mul(q)));

      INDArray oneVector2 = Nd4j.ones(1, 3);
      INDArray delta2 = oneVector2.sub(y2).mul(y2).mul(delta3.mmul(w3));
      b2 = b2.sub(delta2.mul(q));
      w2 = w2.sub(y1.transpose().mmul(delta2.mul(q)));

      INDArray oneVector3 = Nd4j.ones(1, 3);
      INDArray delta1 = oneVector3.sub(y1).mul(y1).mul(delta2.mmul(w2));
      b1 = b1.sub(delta1.mul(q));
      w1 = w1.sub(in.transpose().mmul(delta1.mul(q)));
    }
//    plot(trainings, teachers, w1, b1, w2, b2);
  }

  public static void plot(INDArray trainings, INDArray teachers, INDArray w1, INDArray b1, INDArray w2, INDArray b2, INDArray w3, INDArray b3) {
    double z = 0;
    for(int i = 0; i < 4; i++) {
      INDArray in = trainings.getRow(i);
      INDArray teacher = teachers.getRow(i);

      INDArray y1 = Transforms.sigmoid(in.mmul(w1).addRowVector(b1));
      INDArray y2 = Transforms.sigmoid(y1.mmul(w2).addRowVector(b2));
      INDArray y3 = Transforms.sigmoid(y2.mmul(w3).addRowVector(b3));
      z += teacher.transpose().sub(y3).norm2Number().doubleValue();

      System.out.println(in + " => " + y3 + " : " + teacher);
    }
    System.out.println("LOSS: " + (z / 4));
//    System.out.println((z / 4));
    System.out.println();
  }
}

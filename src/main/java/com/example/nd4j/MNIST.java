package com.example.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.zip.GZIPInputStream;

public class MNIST {
  private static final String TRAIN_IMAGE_FILE = "train-images-idx3-ubyte.gz";
  private static final String TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz";
  private static final String TEST_IMAGE_FILE = "t10k-images-idx3-ubyte.gz";
  private static final String TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz";
  private static final String BASE_PATH = "./dataset/mnist/";
  private static final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";

  private int numImages;
  private int numDimensions;
  private INDArray features;
  private INDArray labels;

  public enum DataType { TRAIN, TEST }

  public MNIST(DataType dataType) throws IOException {
    initialize(dataType);
  }

  private void initialize(DataType dataType) throws IOException {
    String imageFile;
    String labelFile;

    if(DataType.TRAIN == dataType) {
      imageFile = TRAIN_IMAGE_FILE;
      labelFile = TRAIN_LABEL_FILE;
    } else if(DataType.TEST == dataType) {
      imageFile = TEST_IMAGE_FILE;
      labelFile = TEST_LABEL_FILE;
    } else {
      throw new IllegalArgumentException("dataType is invalid.");
    }

    File baseDir = new File(BASE_PATH);
    if (!baseDir.exists()) {
      baseDir.mkdirs();
    }

    download(BASE_URL, BASE_PATH, imageFile);
    download(BASE_URL, BASE_PATH, labelFile);

    this.loadFeatures(imageFile);
    this.loadLabels(labelFile);
  }

  public int getNumImages() {
    return numImages;
  }

  public int getNumDimensions() {
    return numDimensions;
  }

  public INDArray getFeatures() {
    return features;
  }

  public INDArray getFeatures(int... rows) {
    return features.getRows(rows);
  }

  public INDArray getLabels(int... rows) {
    return labels.getRows(rows);
  }

  private void loadFeatures(String imageFile) throws IOException {
    System.out.println("Loading feature data from " + imageFile + " ...");
    try (DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(BASE_PATH + imageFile)))) {
      is.readInt();
      numImages = is.readInt();
      numDimensions = is.readInt() * is.readInt();
      double[][] rawFeatures = new double[numImages][numDimensions];

      for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numDimensions; j++) {
          rawFeatures[i][j] = (double) is.readUnsignedByte() / 255.0;
        }
      }

      features = Nd4j.create(rawFeatures);
    }
  }

  private void loadLabels(String labelFile) throws IOException {
    System.out.println("Loading label data from " + labelFile + " ...");
    try (DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(BASE_PATH + labelFile)))) {
      is.readInt();
      int numLabels = is.readInt();

      double[][] rawLabels = new double[numLabels][10];
      for (int i = 0; i < numLabels; i++) {
        rawLabels[i][is.readUnsignedByte()] = 1.0;
      }

      labels = Nd4j.create(rawLabels);
    }
  }

  private static void download(String baseUrl, String basePath, String fileName) throws IOException {
    if (!new File(basePath + fileName).exists()) {
      System.out.println("Downloading " + baseUrl + fileName + " ...");
      URL url = new URL(baseUrl + fileName);
      URLConnection conn = url.openConnection();
      File file = new File(basePath + fileName);
      try (InputStream in = conn.getInputStream(); FileOutputStream out = new FileOutputStream(file, false)) {
        byte[] data = new byte[1024];
        while (true) {
          int ret = in.read(data);
          if (ret == -1) {
            break;
          }
          out.write(data, 0, ret);
        }
      }
    }
  }
}

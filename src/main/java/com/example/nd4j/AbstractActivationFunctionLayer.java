package com.example.nd4j;

public abstract class AbstractActivationFunctionLayer implements ActivationFunctionLayer {
  public AbstractActivationFunctionLayer(Object instanceToken) {
    if(isEnableToken(instanceToken)) throw new RuntimeException("cannot create instance.");
  }
}

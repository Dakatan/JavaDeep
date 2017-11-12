package com.example.nd4j;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

public interface ActivationFunctionLayer extends Layer {

  default boolean isEnableToken(Object instanceToken) {
    return factory.INSTANCE_TOKEN == instanceToken;
  }

  Factory factory = new Factory();

  static <T extends ActivationFunctionLayer> T newIntance(Class<T> clazz) {
    return factory.newInstance(clazz);
  }

  class Factory {
    private Object INSTANCE_TOKEN = new Object();

    private <T extends ActivationFunctionLayer> T newInstance(Class<T> clazz) {
      try {
        Constructor<T> constructor = clazz.getConstructor(INSTANCE_TOKEN.getClass());
        return constructor.newInstance();
      } catch (NoSuchMethodException | IllegalAccessException | InstantiationException | InvocationTargetException e) {
        throw new RuntimeException("cannot create instance.", e);
      }
    }
  }
}

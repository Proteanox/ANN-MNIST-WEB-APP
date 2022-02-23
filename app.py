import gradio as gr
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model


model = load_model('mnist_model.h5')


def predict_image(img):
  img_3d=img.reshape(-1,28,28)
  im_resize=img_3d/255.0
  prediction=model.predict(im_resize)
  pred=np.argmax(prediction)
  return pred

iface = gr.Interface(predict_image, inputs="sketchpad", outputs="label")

iface.launch(debug='True', share='True')


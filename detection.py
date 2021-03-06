import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


def detect(frame, model):
        frame_ts = tf.image.resize(frame, [299, 299], method='nearest')
        frame_ts = img_to_array(frame_ts)  # output Numpy-array
        frame_ts = frame_ts.reshape((1, frame_ts.shape[0], frame_ts.shape[1], frame_ts.shape[2]))
        frame_ts = preprocess_input(frame_ts)
        yhat = model.predict(frame_ts)
        label = decode_predictions(yhat, top=1000)
        _, objet, proba = label[0][:][0]
        return objet, proba


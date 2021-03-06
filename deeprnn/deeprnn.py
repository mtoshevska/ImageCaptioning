import tensorflow as tf
from .config import Config
from .dataset import Vocabulary, DataSet
from .model import CaptionGenerator
from keras import backend as K


def generate_deeprnn_captions(image_path):
    config = Config()
    config.phase = 'test'
    config.train_cnn = False
    config.beam_size = 3
    tf.reset_default_graph()
    with tf.Session() as sess:
        vocabulary = Vocabulary(config.vocabulary_size, config.vocabulary_file)
        dataset = DataSet([0], [image_path], config.batch_size)
        model = CaptionGenerator(config)
        model.load(sess, 'models/259999.npy')
        tf.get_default_graph().finalize()
        caption = model.test(sess, dataset, vocabulary)
    tf.reset_default_graph()
    K.clear_session()
    return caption[0]

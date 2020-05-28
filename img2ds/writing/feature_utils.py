import tensorflow as tf


def bytes_feature_list(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.FeatureList(feature=[bytes_feature(value) for value in values])


def float_feature_list(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.FeatureList(feature=[float_feature(value) for value in values])


def int64_feature_list(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.FeatureList(feature=[int64_feature(value) for value in values])


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

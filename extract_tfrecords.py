import logging
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np


def extract_tfrecords(data_root, output_dir, n_splits, shuffle, stratify, seed=None):
    image_paths, labels = _extract_image_paths_and_labels(data_root)
    n_images = image_paths.shape[0]
    logging.info(f"{n_images} images have been found.")
    if n_images > 0:
        _write_dataset(image_paths, labels, output_dir=output_dir, n_splits=n_splits, shuffle=shuffle,
                       stratify=stratify, seed=seed)


def _extract_image_paths_and_labels(data_root):
    image_paths_and_labels = pd.DataFrame(columns=['image_path', 'label'])
    for class_folder in os.scandir(data_root):
        if class_folder.is_dir():
            label = class_folder.name
            for image_path in os.scandir(class_folder.path):
                if image_path.is_file():
                    image_path = image_path.path
                    try:
                        Image.open(image_path)
                        image_paths_and_labels = image_paths_and_labels\
                            .append({'image_path': image_path, 'label': label}, ignore_index=True, sort=True)
                    except:
                        pass
    image_paths = image_paths_and_labels['image_path'].values
    labels = image_paths_and_labels['label'].values
    return image_paths, labels


def _create_splitter(n_splits, shuffle, stratify, seed):

    if stratify:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    else:
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)


def _write_dataset(image_paths, labels, output_dir, n_splits, shuffle, stratify, seed=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if n_splits > 1:
        splitter = _create_splitter(n_splits=n_splits, shuffle=shuffle, stratify=stratify, seed=seed)
        for p, (_, indices) in enumerate(splitter.split(image_paths, labels)):
            logging.info(f"Writing data partition #{p+1}...")
            _write_dataset_partition(image_paths=image_paths, labels=labels, indices=indices,
                                     output_path=os.path.join(output_dir, f"data_part_{p + 1}.tfrecords"))
    else:
        logging.info(f"Writing all data into one partition...")
        np.random.seed(seed)
        indices = np.random.choice(np.arange(len(image_paths)), len(image_paths), replace=False)
        _write_dataset_partition(image_paths=image_paths, labels=labels, indices=indices,
                                 output_path=os.path.join(output_dir, "data.tfrecords"))


def _write_dataset_partition(image_paths, labels, indices, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for i in indices:
            image_i = Image.open(image_paths[i])
            example = _serialize_example(image_i, labels[i])
            writer.write(example)
        logging.info(f"{len(indices)} items have been written.")


def _serialize_example(image, label):
    """
     Creates a tf.Example message ready to be written to a file.
     """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    height = image.height
    width = image.width
    depth = len(image.getbands())
    image_bytes = image.tobytes()
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth),
        'label': _bytes_feature(tf.compat.as_bytes(label)),
        'image_raw': _bytes_feature(image_bytes)
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

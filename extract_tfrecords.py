import logging
import os
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from PIL import Image
import pandas as pd


def partition_data(data_root, output_dir, n_splits, seed=None):
    image_paths, labels = _extract_image_paths_and_labels(data_root)
    n_images = image_paths.shape[0]
    logging.info(f"{n_images} images found")
    if n_images > 0:
        _write_dataset_splits(image_paths, labels, output_dir=output_dir, n_splits=n_splits, seed=seed)


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


def _write_dataset_splits(image_paths, labels, output_dir, n_splits, seed=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for p, (_, indices) in enumerate(splitter.split(image_paths, labels)):
        with tf.io.TFRecordWriter(os.path.join(output_dir, f"part_{p+1}.tf")) as writer:
            for i in indices:
                image_i = Image.open(image_paths[i])
                example = _serialize_example(image_i, labels[i])
                writer.write(example)
        logging.info(f"part #{p+1}: {len(indices)} indices")


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

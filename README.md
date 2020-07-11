# Images to dataset
A tool for building a CSV or a Tensorflow dataset from a set of images.

## Installation
1. Install `python3`, `python3-pip` and `wheel`;

2. Move into the root folder of the project;

3. Compile the package:
    ```
    python3 setup.py bdist_wheel
    ```
4. Install on your local machine:
    ```
    pip3 install -U dist/images_to_dataset-*-py3-none-any.whl
    ```

## Usage example
```
python images_to_dataset \
    -input_data_root "images" \
    -output_dir "dataset" \
    -output_file_name "data" \
    -output_format "tfrecords" \
    -n_splits 10 \ # OPTIONAL
    -shuffle \ # OPTIONAL
    -stratify \ # OPTIONAL
    -group "group" \ # OPTIONAL
    -sequence \ # OPTIONAL
    -grid \ # OPTIONAL
    -metadata "metadata.csv" \ # OPTIONAL
    -sample_id "sample_id" \ # OPTIONAL
    -path_column "path" \ # OPTIONAL
    -label_column "label" \ # OPTIONAL
    -additional_columns "col_x" "col_y" ... "col_t" \ # OPTIONAL
    -seed 3 # OPTIONAL
```
where: 
- `input_data_root` is the path of a root directory 
that contains a subdirectory of images for each image label;
- `output_format`: "tfrecords" or "csv"
- `n_splits` is the number of partitions in which the dataset has to be splitted;
- `shuffle` random samples the images (without replacement);
- `stratify` keeps the same proportion of images by label for each partition;
- `group` keeps the samples belonging to the same group in the same partition;
- `sequence` groups the images belonging to the same sample in the same record;
- `grid` groups the images belonging to the same sample in the same record, keeping a grid arrangement;
- `metadata` is the path to a CSV file that can specify `{sample_id}`, `{path_column}`, `{label_column}`, and `{group}` information.

The `CSV` output presents the `path` and `label` columns.

## TFRecords reading example
```
import os
from glob import glob
import tensorflow as tf
from matplotlib import pyplot as plt

# Create a dictionary describing the features.
feature_description = {
    'id': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=''),
    'label': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=''),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
}

def decode_image(image, width, height, depth):
    image = tf.io.decode_raw(image, out_type=tf.uint8)
    image = tf.reshape(image, shape=[height, width, depth])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def _parse_image_function(example):
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example, feature_description)
    image = decode_image(features['image_raw'], 
                         width=features['width'], 
                         height=features['height'], 
                         depth=features['depth'])
    label = tf.squeeze(features['label'])
    return image, label

tfrecord_file_paths = glob(os.path.join('tfrecords', 'data*.tfrecords'))
raw_image_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image, label in parsed_image_dataset.take(5):
    image = image.numpy()
    label = label.numpy().decode("utf-8")
    plt.figure()
    plt.imshow(image)
    plt.title(f'"{label}" sample')
```

## Sequential TFRecords reading example
```
import os
from glob import glob
import tensorflow as tf
from matplotlib import pyplot as plt

# Create a dictionary describing the features.
context_features_description = {
    'id': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=''),
    'label': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=''),
}

sequential_features_description = {
    'height': tf.io.FixedLenSequenceFeature([], tf.int64),
    'width': tf.io.FixedLenSequenceFeature([], tf.int64),
    'depth': tf.io.FixedLenSequenceFeature([], tf.int64),
    'image_raw': tf.io.FixedLenSequenceFeature([], tf.string),
}

# For example we want a final image resolution of 500x500
TARGET_HEIGHT, TARGET_WIDTH = 500, 500

def decode_image(image, width, height, depth):
    image = tf.io.decode_raw(image, out_type=tf.uint8)
    image = tf.reshape(image, shape=[height, width, depth])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, TARGET_HEIGHT, TARGET_WIDTH)
    return image

def _parse_sequence_function(example):
    context_features, sequential_features = \
        tf.io.parse_single_sequence_example(example, context_features=context_features_description, 
                                            sequence_features=sequential_features_description)

    sequence_len = tf.shape(sequential_features['image_raw'])[0]
    images = tf.map_fn(lambda i: decode_image(sequential_features['image_raw'][i], 
                                              width=sequential_features['width'][i], 
                                              height=sequential_features['height'][i], 
                                              depth=sequential_features['depth'][i]),
                       elems=tf.range(sequence_len), dtype=tf.float32)

    label = tf.squeeze(context_features['label'])
    return images, label

tfrecord_file_paths = glob(os.path.join('tfrecords', 'data*.tfrecords'))
raw_image_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
parsed_seq_image_dataset = raw_image_dataset.map(_parse_sequence_function)

for i, (image, label) in enumerate(parsed_seq_image_dataset.take(5)):
    label = label.numpy().decode('UTF-8')
    for j in range(image.shape[0]):
        plt.figure()
        plt.imshow(image[j])
        plt.title(f'"{label}" example {i} part {j}')
```

## Dependencies
- Python 3.6.9
- Tensorflow 2.1.0
- scikit-learn 0.22.1
- Pillow 6.2.1
- Pandas 1.0.1
- Numpy 1.18.1
- Matplotlib 3.2.0

# Images to tfrecords
A tool for building a Tensorflow dataset from a set of images.

## Usage example
```
python images_to_tfrecords \
    -data_root "images" \
    -output_dir "tfrecords" \
    -n_splits 10 \ # OPTIONAL
    -shuffle \ # OPTIONAL
    -stratify \ # OPTIONAL
    -seed 3 # OPTIONAL
```
where: 
- `data_root` is the path of a root directory 
that contains a subdirectory of images for each image label;
- `n_splits` is the number of partitions in which the dataset has to be splitted;
- `shuffle` random samples the images (without replacement);
- `stratify` keeps the same proportion of images by label for each partition.

## TFRecords reading example
```
import os
from glob import glob
import tensorflow as tf
from matplotlib import pyplot as plt

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=''),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}

def decode_image(image, width, height, depth):
    image = tf.io.decode_raw(image, out_type=tf.uint8)
    image = tf.reshape(image, shape=[height, width, depth])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = decode_image(features['image_raw'], 
                         width=features['width'], 
                         height=features['height'], 
                         depth=features['depth'])
    label = tf.squeeze(features['label'])
    return image, label

tfrecord_file_paths = glob(os.path.join('images', 'tfrecords', 'data*.tfrecords'))
raw_image_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image, label in parsed_image_dataset.take(5):
  image = image.numpy()
  label = label.numpy().decode("utf-8")
  plt.figure()
  plt.imshow(image)
  plt.title(f'"{label}" sample')
```

## Dependencies
- Python 3.6.9
- Tensorflow 2.1.0
- scikit-learn 0.22.1
- Pillow 6.2.1
- Pandas 1.0.1
- Numpy 1.18.1
- Matplotlib 3.2.0

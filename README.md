# Images to tfrecords
A tool for building a Tensorflow dataset from a set of images.

## Use
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

## Dependencies
- Python 3.6.9
- Tensorflow 2.1.0
- scikit-learn 0.22.1
- Pillow 6.2.1
- Pandas 1.0.1
- Numpy 1.18.1

import argparse
import logging
from pathlib import Path

from image_dataset_builder import ImageDatasetBuilder
from image_tfrecords_writer import ImageTFRecordsWriter

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("-data_root", type=Path, required=True)
parser.add_argument("-output_dir", type=Path, required=True, help="path to the output directory")
parser.add_argument("-n_splits", type=int, required=False, default=1)
parser.add_argument("-shuffle", action="store_true")
parser.add_argument("-stratify", action="store_true")
parser.add_argument("-seed", type=int, required=False)
args = parser.parse_args()

n_splits = args.n_splits

partitions = ImageDatasetBuilder(data_root=args.data_root, n_splits=n_splits,
                                 shuffle=args.shuffle, stratify=args.stratify, seed=args.seed).build()

ImageTFRecordsWriter(n_splits=n_splits, output_dir=args.output_dir).write(partitions)

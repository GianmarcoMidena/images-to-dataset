import argparse
import logging
from pathlib import Path

from image_dataset_builder import ImageDatasetBuilder
from image_tfrecords_writer import ImageTFRecordsWriter

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("-input_data_root", type=Path, required=False)
parser.add_argument("-output_dir", type=Path, required=True, help="path to the output directory")
parser.add_argument("-n_splits", type=int, required=False, default=1)
parser.add_argument("-shuffle", action="store_true")
parser.add_argument("-stratify", action="store_true")
parser.add_argument("-group", type=str, required=False, help="group name")
parser.add_argument("-metadata", type=Path)
parser.add_argument("-path_column", type=str, default="path")
parser.add_argument("-label_column", type=str, default="label")
parser.add_argument("-seed", type=int, required=False)
args = parser.parse_args()

n_splits = args.n_splits

partitions = ImageDatasetBuilder(data_root=args.input_data_root, n_splits=n_splits,
                                 with_shuffle=args.shuffle, with_stratify=args.stratify, group=args.group,
                                 metadata=args.metadata, path_column=args.path_column, label_column=args.label_column,
                                 seed=args.seed).build()

ImageTFRecordsWriter(n_splits=n_splits, output_dir=args.output_dir).write(partitions)

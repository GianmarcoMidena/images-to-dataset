import argparse
import logging
from pathlib import Path

from img2ds.building import ImageDatasetBuilder
from img2ds.writing import CSVWriter
from img2ds.writing import ImageTFRecordsWriter
from img2ds.writing import SequenceImageTFRecordsWriter
from img2ds.writing.grid_image_tfrecords_writer import GridImageTFRecordsWriter

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("-input_data_root", type=Path, required=False)
parser.add_argument("-input_archive_path", type=Path, required=False)
parser.add_argument("-output_dir", type=Path, required=True, help="path to the output directory")
parser.add_argument("-output_file_name", type=str, default="data")
parser.add_argument("-output_format", type=str, choices=['csv', 'tfrecords'], required=True)
parser.add_argument("-n_splits", type=int, required=False, default=1)
parser.add_argument("-shuffle", action="store_true")
parser.add_argument("-stratify", action="store_true")
parser.add_argument("-group", type=str, required=False, help="group name")
parser.add_argument("-sequence", action="store_true")
parser.add_argument("-grid", action="store_true")
parser.add_argument("-sample_id", type=str, required=False)
parser.add_argument("-metadata", type=Path, required=False)
parser.add_argument("-path_column", type=str, default="path")
parser.add_argument("-label_column", type=str, default="label")
parser.add_argument("-global_label_column", type=str, required=False)
parser.add_argument("-local_label_column", type=str, required=False)
parser.add_argument("-additional_columns", action="append", required=False)
parser.add_argument("-seed", type=int, required=False)
args = parser.parse_args()

n_splits = args.n_splits

partitions = ImageDatasetBuilder(data_root=args.input_data_root, n_splits=n_splits,
                                 with_shuffle=args.shuffle, with_stratify=args.stratify, group=args.group,
                                 metadata=args.metadata, path_column=args.path_column,
                                 label_column=args.label_column,
                                 local_label_column=args.local_label_column,
                                 global_label_column=args.global_label_column,
                                 sequence_or_grid=args.sequence or args.grid, sample_id=args.sample_id,
                                 additional_columns=args.additional_columns, seed=args.seed).build()

if args.output_format == 'csv':
    CSVWriter(n_splits=n_splits, output_dir=args.output_dir, output_file_name=args.output_file_name).write(partitions)
elif args.sequence:
    SequenceImageTFRecordsWriter(n_splits=n_splits, output_dir=args.output_dir,
                                 output_file_name=args.output_file_name).write(partitions)
elif args.grid:
    GridImageTFRecordsWriter(n_splits=n_splits, output_dir=args.output_dir,
                             output_file_name=args.output_file_name).write(partitions)
else:
    ImageTFRecordsWriter(n_splits=n_splits, output_dir=args.output_dir,
                         output_file_name=args.output_file_name,
                         input_archive_path=args.input_archive_path).write(partitions)

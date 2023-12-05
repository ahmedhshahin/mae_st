# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from pathlib import Path

from main_pretrain import get_args_parser, main
from util.dataset import CTData


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # save data to tmpfs
    _ = CTData(
        args.path_to_data_dir,
        args.path_to_df,
        mode="train",
        num_slices=args.num_slices,
        n=args.n,
        store_data_to_tmpfs=True,
        seed=args.seed,
    )
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

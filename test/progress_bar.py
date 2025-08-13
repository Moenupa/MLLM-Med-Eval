import json
import os.path as osp
from glob import glob

import pandas as pd

SRC_LINES = {
    osp.basename(fp).split(".")[0]: len(json.load(open(fp)))
    for fp in glob("output/*.json")
}

for target_file in glob("output*/.cache_gpt_metrics/*"):
    filename = osp.basename(target_file).split(".")[0]
    src_lines = SRC_LINES[filename]
    df = pd.read_json(target_file, lines=True)

    target_lines = df["key"].nunique()
    if target_lines != src_lines:
        print(f"{target_lines:04d}/{src_lines:04d}\t{target_file}")

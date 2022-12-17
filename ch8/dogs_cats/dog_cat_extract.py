import os, shutil, pathlib

data_dir = pathlib.Path("dogs-vs-cats")
orig_dir = data_dir / "train"
new_base_dir = data_dir / "cats_vs_dogs_small"

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg"\
            for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=orig_dir / fname, dst=dir / fname)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)


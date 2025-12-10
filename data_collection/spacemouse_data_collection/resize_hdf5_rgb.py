"""
Resize RGB frames in data.hdf5 to a target resolution and write back to HDF5.

python resize_hdf5_rgb.py \
    --input /home/le/ARX_X5/datasets/pick/episode_0/data.hdf5 \
    --height 240 --width 320
"""
import argparse
import os
from typing import Iterable, Optional

import cv2
import h5py
import numpy as np

def iter_rgb_frames(rgb_group) -> Iterable[np.ndarray]:
    """Iterate frames under observation/rgb in order."""
    # Keys are "0", "1", ...; sort numerically to keep order.
    for key in sorted(rgb_group.keys(), key=lambda k: int(k)):
        yield np.array(rgb_group[key])


def resize_frames(rgb_group, height: int, width: int):
    """Return resized frames while preserving single/multi-camera shape."""
    resized_list = []
    for frame in iter_rgb_frames(rgb_group):
        # frame can be [cam, H, W, 3] or [H, W, 3]
        if frame.ndim == 4:
            cams = frame.shape[0]
            new_frames = []
            for cam_idx in range(cams):
                rgb = frame[cam_idx]
                resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                new_frames.append(resized)
            resized = np.stack(new_frames, axis=0)
        elif frame.ndim == 3:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        else:
            raise ValueError(f"不支持的帧维度: {frame.shape}")
        resized_list.append(resized)
    return resized_list


def overwrite_rgb_group(h5file: h5py.File, resized_frames):
    """Remove old observation/rgb and write resized frames with compression."""
    obs_group = h5file["observation"]
    if "rgb" in obs_group:
        del obs_group["rgb"]
    rgb_group = obs_group.create_group("rgb")
    for idx, frame in enumerate(resized_frames):
        # Align with process_endpose: use compression level 3 (gzip here).
        rgb_group.create_dataset(
            str(idx),
            data=frame,
            compression="gzip",
            compression_opts=3,
        )


def main():
    parser = argparse.ArgumentParser(description="Resize RGB frames in data.hdf5 and write back")
    parser.add_argument("--input", required=True, help="Input HDF5 path, e.g., .../episode_x/data.hdf5")
    parser.add_argument("--output", help="Output HDF5 path, default overwrites input")
    parser.add_argument("--height", type=int, default=240, help="Target height, default 240")
    parser.add_argument("--width", type=int, default=320, help="Target width, default 320")
    args = parser.parse_args()

    output_path: Optional[str] = args.output or args.input

    if output_path != args.input:
        import shutil

        shutil.copy2(args.input, output_path)

    with h5py.File(output_path, "r+") as f:
        if "observation" not in f or "rgb" not in f["observation"]:
            raise KeyError("observation/rgb not found in HDF5 file")
        rgb_group = f["observation"]["rgb"]
        resized_frames = resize_frames(rgb_group, args.height, args.width)
        overwrite_rgb_group(f, resized_frames)
    print(f"Finished resizing and writing back: {output_path}")


if __name__ == "__main__":
    main()


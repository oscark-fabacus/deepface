#!/usr/bin/env python3
"""Extract faces from an image using DeepFace RetinaFace backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
from deepface import DeepFace


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def iter_image_paths(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in IMAGE_EXTENSIONS:
                    yield child
        else:
            yield path


def normalize_face(face) -> "cv2.typing.MatLike":
    if face.dtype == "uint8":
        return face
    if face.max() <= 1.0:
        face = (face * 255).clip(0, 255)
    return face.astype("uint8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract faces from images and save crops using RetinaFace."
    )
    parser.add_argument(
        "image_paths",
        nargs="+",
        help="Path(s) to images or a directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        default="face_crops",
        help="Directory to save extracted face images.",
    )
    parser.add_argument(
        "--enforce-detection",
        action="store_true",
        default=False,
        help="Raise an error if no face is detected.",
    )
    parser.add_argument(
        "--expand-percentage",
        type=int,
        default=0,
        help="Expand detected face region by a percentage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_image_paths(Path(path) for path in args.image_paths):
        faces = DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend="retinaface",
            enforce_detection=args.enforce_detection,
            align=True,
            expand_percentage=args.expand_percentage,
            color_face="bgr",
            normalize_face=False,
        )
        for idx, face_obj in enumerate(faces, start=1):
            face = normalize_face(face_obj["face"])
            area = face_obj["facial_area"]
            output_name = (
                f"{image_path.stem}_face{idx}_x{area['x']}_y{area['y']}"
                f"_w{area['w']}_h{area['h']}{image_path.suffix}"
            )
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), face)


if __name__ == "__main__":
    main()

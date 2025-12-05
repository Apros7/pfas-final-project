import argparse
from pathlib import Path

from constants import YOLO_MODEL_PATH
from runner import Runner
from viewer import Viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the viewer on a given KITTI sequence.")
    parser.add_argument(
        "--seq-dir",
        type=str,
        required=True,
        help="Path to the sequence folder (e.g. 34759_final_project_rect/seq_01)",
    )
    parser.add_argument("--model", type=str, default=YOLO_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Force model device (cpu/cuda)")
    parser.add_argument("--wait", type=int, default=1, help="Viewer waitKey delay in ms")
    parser.add_argument("--loop", action="store_true", help="Loop through frames continuously")
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to save the rendered viewer video. "
            "Use --save without a value to save into <seq-dir>/viewer_output.mp4"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = args.save
    if save_path == "":
        save_path = str(Path(args.seq_dir) / "viewer_output.mp4")
    viewer = Viewer(args.seq_dir)
    painter = Runner(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        seq_dir=args.seq_dir,
        fx=viewer.cameras[0].intrinsics.fx if viewer.cameras[0].intrinsics else None,
        fy=viewer.cameras[0].intrinsics.fy if viewer.cameras[0].intrinsics else None,
        cx=viewer.cameras[0].intrinsics.cx if viewer.cameras[0].intrinsics else None,
        cy=viewer.cameras[0].intrinsics.cy if viewer.cameras[0].intrinsics else None,
        baseline=viewer.baseline_m,
    )
    viewer.run(painter, wait_ms=args.wait, loop=args.loop, save_path=save_path)


if __name__ == "__main__":
    main()


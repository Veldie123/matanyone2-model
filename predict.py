"""
Replicate Cog predictor for MatAnyone 2 AI video matting.

Takes a greenscreen video + optional first-frame mask and produces
a ProRes 4444 RGBA video with foreground + alpha merged.

GitHub: https://github.com/pq-yang/MatAnyone2
HuggingFace: AEmotionStudio/matanyone2
"""

import os
import subprocess
import sys
import tempfile
import time

from cog import BasePredictor, Input, Path

# Add parent dir for mask_generator import
sys.path.insert(0, os.path.dirname(__file__))


class Predictor(BasePredictor):
    def setup(self):
        """Load MatAnyone 2 model — runs once when container starts."""
        from matanyone2 import MatAnyone2, InferenceCore

        self.model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2")
        self.processor = InferenceCore(self.model, device="cuda:0")

    def predict(
        self,
        video: Path = Input(
            description="Greenscreen video file (MP4/MOV)"
        ),
        mask: Path = Input(
            description=(
                "Optional first-frame binary mask PNG "
                "(white=foreground, black=background). "
                "If not provided, auto-generated from greenscreen."
            ),
            default=None,
        ),
        lower_green_h: int = Input(
            description="HSV lower hue for green detection",
            default=35,
        ),
        upper_green_h: int = Input(
            description="HSV upper hue for green detection",
            default=85,
        ),
    ) -> Path:
        """Run MatAnyone 2 video matting and return foreground with alpha."""

        with tempfile.TemporaryDirectory() as workdir:
            # IMPORTANT: Cog downloads inputs to files WITHOUT extension
            # (e.g. /tmp/tmpXXXXdownload). MatAnyone 2's ffmpeg call fails
            # without a proper extension. Copy to a renamed file first.
            input_path = os.path.join(workdir, "input.mp4")
            subprocess.run(["cp", str(video), input_path], check=True)

            # Generate mask if not provided
            if mask is None:
                mask_path = os.path.join(workdir, "mask.png")
                self._generate_greenscreen_mask(
                    input_path, mask_path, lower_green_h, upper_green_h
                )
            else:
                # Copy mask too (same issue with extensions)
                mask_path = os.path.join(workdir, "mask.png")
                subprocess.run(["cp", str(mask), mask_path], check=True)

            # Run inference — returns tuple (foreground_path, alpha_path)
            output_dir = os.path.join(workdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            start = time.time()
            result = self.processor.process_video(
                input_path=input_path,
                mask_path=mask_path,
                output_path=output_dir,
            )
            elapsed = time.time() - start
            print(f"Inference completed in {elapsed:.1f}s")
            print(f"process_video returned: {result}")

            # process_video returns a tuple (foreground_path, alpha_path)
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                actual_fg = str(result[0])
                actual_alpha = str(result[1])
            else:
                # Fallback: search the output directory
                print(f"Unexpected return type, searching output_dir")
                output_files = os.listdir(output_dir)
                print(f"Output files: {output_files}")
                fg_candidates = [f for f in output_files if any(k in f.lower() for k in ["foreground", "fg", "com"])]
                alpha_candidates = [f for f in output_files if any(k in f.lower() for k in ["alpha", "pha", "matte"])]
                if not fg_candidates or not alpha_candidates:
                    raise RuntimeError(f"Could not find foreground/alpha outputs. Files: {output_files}")
                actual_fg = os.path.join(output_dir, fg_candidates[0])
                actual_alpha = os.path.join(output_dir, alpha_candidates[0])

            print(f"Foreground: {actual_fg}")
            print(f"Alpha: {actual_alpha}")

            # Merge RGB foreground + grayscale alpha into ProRes 4444 RGBA
            merged_path = "/tmp/matted_output.mov"
            cmd = [
                "ffmpeg", "-y",
                "-i", actual_fg,
                "-i", actual_alpha,
                "-filter_complex", "[0:v][1:v]alphamerge[out]",
                "-map", "[out]",
                "-c:v", "prores_ks", "-profile:v", "4",
                "-pix_fmt", "yuva444p10le",
                "-an",
                merged_path,
            ]
            merge_result = subprocess.run(cmd, capture_output=True, text=True)
            if merge_result.returncode != 0:
                raise RuntimeError(f"FFmpeg merge failed: {merge_result.stderr[-500:]}")

            return Path(merged_path)

    def _generate_greenscreen_mask(
        self,
        video_path: str,
        output_path: str,
        lower_h: int = 35,
        upper_h: int = 85,
    ) -> None:
        """Generate binary mask from greenscreen video's first frame."""
        import cv2
        import numpy as np

        # Extract first frame
        frame_path = output_path.replace(".png", "_frame.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "1",
                frame_path,
            ],
            capture_output=True,
            check=True,
        )

        frame = cv2.imread(frame_path)
        if frame is None:
            raise RuntimeError(
                f"Failed to read frame from {video_path}"
            )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([lower_h, 50, 50], dtype=np.uint8)
        upper = np.array([upper_h, 255, 255], dtype=np.uint8)

        green_mask = cv2.inRange(hsv, lower, upper)
        fg_mask = cv2.bitwise_not(green_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        cv2.imwrite(output_path, fg_mask)
        os.remove(frame_path)

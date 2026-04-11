"""
scripts/run_pipeline.py
=======================
Real-time BCI pipeline — end-to-end simulation.

Data Flow:
  MockEEGStream (training .gdf)
       ↓  read(STEP_SAMPLES)
  SlidingWindowExtractor
       ↓  returns window every STEP_SIZE_SEC
  EEGPredictor
       ↓  raw Left / Right probabilities
  DecisionSmoother
       ↓  stable "LEFT" | "RIGHT" | "STOP"
  ControlPolicy  ← ACTION_LOOKUP (lookup table)
       ↓  "MOVE_LEFT" | "MOVE_RIGHT" | "HOLD"
  Terminal Display (with ground-truth label from training annotations)

Ground-Truth Labels:
  The BCI-2a TRAINING files contain event annotations that tell us what
  motor imagery the subject was *actually* performing at each moment.
  We display this alongside the model's prediction so you can visually
  assess when the system is correct.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --file data/BCI-2A/A03T.gdf
  python scripts/run_pipeline.py --no-ica   (skip ICA for fast testing)

Press Ctrl+C to stop.
"""

import sys
import time
import argparse
from pathlib import Path

# Make src/ importable when running from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import (
    MODEL_PATH,
    DATA_DIR,
    STEP_SAMPLES,
    STEP_SIZE_SEC,
    WINDOW_SAMPLES,
    WINDOW_SIZE_SEC,
    SAMPLING_RATE,
    N_CLASSES_CHECKPOINT,
    ACTIVE_CLASSES,
    CLASS_TO_LABEL,
)
from src.streaming.mock_stream import MockEEGStream
from src.inference.window import SlidingWindowExtractor
from src.inference.predictor import EEGPredictor
from src.policy.smoother import DecisionSmoother
from src.policy.control import ControlPolicy, ACTION_LOOKUP


# ── ANSI colour codes for terminal output ─────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ACTION_COLOUR = {
    "MOVE_LEFT":  CYAN,
    "MOVE_RIGHT": GREEN,
    "HOLD":       GREY,
}


def colour_action(action: str) -> str:
    c = ACTION_COLOUR.get(action, RESET)
    return f"{c}{action:<12}{RESET}"


def colour_match(pred_label: str, true_label: str | None) -> str:
    """Colour the true label green if it matches the prediction, red if not."""
    if true_label is None:
        return f"{GREY}{'—':>10}{RESET}"
    if pred_label == true_label:
        return f"{GREEN}{true_label:>10}{RESET}"
    return f"{RED}{true_label:>10}{RESET}"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BCI Real-Time Pipeline")
    p.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a BCI-2a training .gdf file (default: first A??T.gdf found)",
    )
    p.add_argument(
        "--no-ica",
        action="store_true",
        help="Skip ICA preprocessing (faster startup, slightly lower quality)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the trained model .pth file",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (2.0 = 2× faster than real-time, 0 = no sleep)",
    )
    return p.parse_args()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    # ── Resolve GDF file ──────────────────────────────────────────────────────
    if args.file:
        gdf_path = Path(args.file)
    else:
        candidates = sorted(DATA_DIR.glob("*T.gdf"))
        if not candidates:
            print(f"[ERROR] No *T.gdf files found in {DATA_DIR}")
            sys.exit(1)
        gdf_path = candidates[0]

    apply_ica = not args.no_ica
    # speed=0 → no sleep (run as fast as possible)
    sleep_sec = 0.0 if args.speed == 0 else (STEP_SIZE_SEC / args.speed)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  BCI Real-Time Pipeline  --  Motor Imagery -> Robot Command{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"  Data file  : {gdf_path.name}")
    print(f"  Model      : {Path(args.model).name}")
    print(f"  Window     : {WINDOW_SIZE_SEC:.1f} s  ({WINDOW_SAMPLES} samples)")
    print(f"  Step       : {STEP_SIZE_SEC:.2f} s  ({STEP_SAMPLES} samples)")
    print(f"  Speed      : {args.speed:.1f}×  (sleep {sleep_sec*1000:.0f} ms/step)")
    print(f"  ICA        : {'yes' if apply_ica else 'no (--no-ica)'}")
    print()

    # ── Explain the lookup table once at startup ──────────────────────────────
    print(f"{BOLD}  Lookup Table  (Control Policy):{RESET}")
    for dec, act in ACTION_LOOKUP.items():
        print(f"    {dec:<8} → {act}")
    print()

    # ── Initialise modules ────────────────────────────────────────────────────
    stream    = MockEEGStream(str(gdf_path), apply_ica=apply_ica)
    extractor = SlidingWindowExtractor(WINDOW_SAMPLES, STEP_SAMPLES)
    predictor = EEGPredictor(args.model, n_channels=stream.n_channels)
    smoother  = DecisionSmoother()
    policy    = ControlPolicy()

    # ── Column headers ────────────────────────────────────────────────────────
    print(f"\n{BOLD}"
          f"{'Step':>6}  "
          f"{'P(Left)':>8}  "
          f"{'P(Right)':>8}  "
          f"{'RawPred':>8}  "
          f"{'Smooth':>8}  "
          f"{'Action':<14}"
          f"{'TrueLabel':>10}"
          f"{RESET}")
    print("─" * 76)

    # ── Main loop ─────────────────────────────────────────────────────────────
    step = 0
    try:
        while True:
            t0 = time.perf_counter()

            # 1. Read one step's worth of samples from the stream
            chunk = stream.read(STEP_SAMPLES)          # (n_ch, STEP_SAMPLES)

            # The window START in the file (before the pointer advanced)
            #   pointer just moved forward by STEP_SAMPLES, so:
            window_start = (stream.current_position - WINDOW_SAMPLES) % stream.n_samples

            # 2. Add chunk to the sliding window buffer
            win = extractor.add(chunk)
            if win is None:
                continue    # buffer not yet full — skip display

            # 3. Ground-truth label for this window position
            true_label = stream.get_true_label(window_start)

            # 4. Model inference → raw probabilities
            pred = predictor.predict(win)

            # 5. Decision smoothing → stable decision
            decision = smoother.update(pred)

            # 6.  ── LOOKUP TABLE ──  decision → robot action
            #   ACTION_LOOKUP lives in src/policy/control.py
            #   {"LEFT": "MOVE_LEFT",  "RIGHT": "MOVE_RIGHT",  "STOP": "HOLD"}
            action = policy.get_action(decision)

            # 7. Terminal display
            match_str = colour_match(pred["label"], true_label)
            print(
                f"{step:>6}  "
                f"{pred['left']:>8.3f}  "
                f"{pred['right']:>8.3f}  "
                f"{pred['label']:>8}  "
                f"{decision:>8}  "
                f"{colour_action(action)}"
                f"{match_str}"
            )

            step += 1

            # 8. Pace output to (approximately) real-time
            elapsed = time.perf_counter() - t0
            remaining = sleep_sec - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[Pipeline] Stopped after {step} windows.{RESET}\n")


if __name__ == "__main__":
    run(parse_args())

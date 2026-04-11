"""
scripts/demo_pipeline.py
========================
Clean demonstration pipeline using DemoEEGStream.

Unlike run_pipeline.py (which streams raw continuous EEG with underfined
rest periods), this script streams ONLY actual labeled MI epochs so that:
  - Every window shown has a known ground truth
  - You can see exactly when LEFT vs RIGHT is being presented
  - Trial number, label, and accuracy are tracked precisely

Demo modes:
  --mode left        : stream only LEFT trials  (tests LEFT prediction)
  --mode right       : stream only RIGHT trials (tests RIGHT prediction)
  --mode alternating : LEFT, RIGHT, LEFT, RIGHT ... (balanced, default)
  --mode all         : all trials in dataset order

Usage:
  python scripts/demo_pipeline.py                           # alternating, no ICA
  python scripts/demo_pipeline.py --mode left               # only LEFT trials
  python scripts/demo_pipeline.py --mode right              # only RIGHT trials
  python scripts/demo_pipeline.py --mode alternating --ica  # with ICA (better)
  python scripts/demo_pipeline.py --n-trials 6             # 3 LEFT + 3 RIGHT

Press Ctrl+C to stop and see summary.
"""

import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import (
    MODEL_PATH, DATA_DIR, STEP_SAMPLES, STEP_SIZE_SEC,
    WINDOW_SAMPLES, WINDOW_SIZE_SEC,
)
from src.streaming.demo_stream  import DemoEEGStream
from src.inference.window       import SlidingWindowExtractor
from src.inference.predictor    import EEGPredictor
from src.policy.smoother        import DecisionSmoother
from src.policy.control         import ControlPolicy, ACTION_LOOKUP

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ACTION_COLOUR = {"MOVE_LEFT": CYAN, "MOVE_RIGHT": GREEN, "HOLD": GREY}


def fmt_action(action):
    return f"{ACTION_COLOUR.get(action, RESET)}{action:<12}{RESET}"


def fmt_label(pred, truth):
    if truth is None:
        return f"{GREY}{'[rest]':>9}{RESET}"
    color = GREEN if pred == truth else RED
    mark  = "OK" if pred == truth else "!!"
    return f"{color}{truth:>7} {mark}{RESET}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BCI Demo Pipeline — labeled trials only")
    p.add_argument("--file",      type=str,   default=None,
                   help="Path to *T.gdf training file (default: first found)")
    p.add_argument("--model",     type=str,   default=str(MODEL_PATH))
    p.add_argument("--mode",      type=str,   default="alternating",
                   choices=["left", "right", "alternating", "all"],
                   help="Trial selection mode")
    p.add_argument("--n-trials",  type=int,   default=None,
                   help="Limit to first N trials")
    p.add_argument("--ica",       action="store_true",
                   help="Apply ICA (slower startup, more realistic)")
    p.add_argument("--speed",     type=float, default=0,
                   help="Playback speed multiplier (0 = max speed)")
    p.add_argument("--trials-only", action="store_true",
                   help="Skip rest window rows from display")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    # Resolve file
    if args.file:
        gdf_path = Path(args.file)
    else:
        candidates = sorted(DATA_DIR.glob("*T.gdf"))
        if not candidates:
            print(f"[ERROR] No *T.gdf files found in {DATA_DIR}")
            sys.exit(1)
        gdf_path = candidates[0]

    sleep_sec = 0.0 if args.speed == 0 else (STEP_SIZE_SEC / args.speed)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  BCI Demo Pipeline  --  Labeled Epoch Replay{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"  File     : {gdf_path.name}")
    print(f"  Mode     : {args.mode.upper()}")
    print(f"  N trials : {args.n_trials or 'all'}")
    print(f"  ICA      : {'yes' if args.ica else 'no (use --ica for cleaner signal)'}")
    print(f"  Speed    : {args.speed}x")
    print()
    print(f"{BOLD}  Lookup Table:{RESET}")
    for dec, act in ACTION_LOOKUP.items():
        print(f"    {dec:<8} -> {act}")
    print()
    print(f"{GREY}  Label column: {GREEN}OK{GREY}=correct  {RED}!!{GREY}=wrong  [rest]=between trials{RESET}")
    print()

    # ── Init stream ───────────────────────────────────────────────────────────
    stream = DemoEEGStream(
        gdf_path     = str(gdf_path),
        mode         = args.mode,
        n_trials     = args.n_trials,
        apply_ica    = args.ica,
        verbose      = True,
    )
    extractor = SlidingWindowExtractor(WINDOW_SAMPLES, STEP_SAMPLES)
    predictor = EEGPredictor(args.model, n_channels=stream.n_channels)
    smoother  = DecisionSmoother()
    policy    = ControlPolicy()

    # ── Accuracy trackers ─────────────────────────────────────────────────────
    raw_counts     = defaultdict(lambda: {"correct": 0, "total": 0})
    smooth_counts  = defaultdict(lambda: {"correct": 0, "total": 0})

    # ── Headers ───────────────────────────────────────────────────────────────
    print(f"{BOLD}"
          f"{'Step':>6}  {'Trial':>7}  "
          f"{'P(L)':>7}  {'P(R)':>7}  "
          f"{'RawPred':>8}  {'Smooth':>8}  "
          f"{'Action':<14}"
          f"{'TrueLabel':>9}  "
          f"{'RawAcc':>7}"
          f"{RESET}")
    print("-" * 85)

    step = 0
    try:
        while True:
            t0 = time.perf_counter()

            chunk = stream.read(STEP_SAMPLES)
            window_start = (stream.current_position - WINDOW_SAMPLES) % stream.n_samples

            win = extractor.add(chunk)
            if win is None:
                continue

            true_label = stream.get_true_label(window_start)
            trial_info = stream.get_trial_info(window_start)

            # Model inference
            pred     = predictor.predict(win)
            decision = smoother.update(pred)
            action   = policy.get_action(decision)

            trial_str = ""
            if trial_info:
                trial_str = f"T{trial_info['trial_idx']:>2}/{trial_info['total']}"

            # Track accuracy on labeled windows
            acc_str = ""
            if true_label is not None:
                raw_counts[true_label]["total"]   += 1
                smooth_counts[true_label]["total"] += 1
                if pred["label"] == true_label:
                    raw_counts[true_label]["correct"] += 1
                if decision == true_label:
                    smooth_counts[true_label]["correct"] += 1

                total   = sum(v["total"]   for v in raw_counts.values())
                correct = sum(v["correct"] for v in raw_counts.values())
                acc_str = f"{correct/total:>7.1%}"

            # Skip rest if requested
            if args.trials_only and true_label is None:
                step += 1
                continue

            print(
                f"{step:>6}  {trial_str:>7}  "
                f"{pred['left']:>7.3f}  {pred['right']:>7.3f}  "
                f"{pred['label']:>8}  {decision:>8}  "
                f"{fmt_action(action)}"
                f"{fmt_label(pred['label'], true_label)}  "
                f"{CYAN}{acc_str}{RESET}"
            )

            step += 1

            elapsed = time.perf_counter() - t0
            if sleep_sec - elapsed > 0:
                time.sleep(sleep_sec - elapsed)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopped after {step} steps.{RESET}\n")

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"{BOLD}  Raw Prediction Accuracy (per window):{RESET}")
        print(f"  {'Class':<8}  {'Correct':>8}  {'Total':>8}  {'Acc':>8}")
        print(f"  {'-'*36}")
        for cls in ("LEFT", "RIGHT"):
            d = raw_counts[cls]
            if d["total"] > 0:
                print(f"  {cls:<8}  {d['correct']:>8}  {d['total']:>8}  {d['correct']/d['total']:>8.1%}")
        raw_total   = sum(v["total"]   for v in raw_counts.values())
        raw_correct = sum(v["correct"] for v in raw_counts.values())
        if raw_total > 0:
            print(f"  {'TOTAL':<8}  {raw_correct:>8}  {raw_total:>8}  {raw_correct/raw_total:>8.1%}")

        print(f"\n{BOLD}  Smoothed Decision Accuracy (per window):{RESET}")
        print(f"  {'Class':<8}  {'Correct':>8}  {'Total':>8}  {'Acc':>8}")
        print(f"  {'-'*36}")
        for cls in ("LEFT", "RIGHT"):
            d = smooth_counts[cls]
            if d["total"] > 0:
                print(f"  {cls:<8}  {d['correct']:>8}  {d['total']:>8}  {d['correct']/d['total']:>8.1%}")

        print()
        if raw_counts["LEFT"]["total"] > 0 and raw_counts["RIGHT"]["total"] > 0:
            l_acc = raw_counts["LEFT"]["correct"]  / raw_counts["LEFT"]["total"]
            r_acc = raw_counts["RIGHT"]["correct"] / raw_counts["RIGHT"]["total"]
            if l_acc > r_acc + 0.15:
                print(f"  {YELLOW}Note: LEFT accuracy ({l_acc:.1%}) >> RIGHT ({r_acc:.1%})")
                print(f"  This is a class bias. Retrain with CrossEntropyLoss(weight=[1.0, w])")
                print(f"  where w = (n_left / n_right) to upweight RIGHT samples.{RESET}")
        print()


if __name__ == "__main__":
    run(parse_args())

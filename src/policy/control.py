"""
src/policy/control.py
=====================
ControlPolicy — the "Control Policy Layer" from the BCI pipeline.

┌─────────────────────────────────────────────────────┐
│               THE LOOKUP TABLE                      │
│                                                     │
│  This is where the BCI output becomes a robot CMD.  │
│                                                     │
│   Smoothed Decision │ Robot Action                  │
│   ─────────────────┼──────────────                  │
│   "LEFT"           │ "MOVE_LEFT"                    │
│   "RIGHT"          │ "MOVE_RIGHT"                   │
│   "STOP"           │ "HOLD"                         │
│                                                     │
│  It is applied AFTER DecisionSmoother, so only      │
│  stable, high-confidence decisions reach the robot. │
└─────────────────────────────────────────────────────┘

Dwell Time:
  Even after smoothing, we add a second safety layer: the same
  decision must persist for `dwell_steps` consecutive windows before
  the command is actually dispatched.  Commands are cheap to compute
  but robot motion must only happen on sustained intent.

When PyBullet is added, `robot.step(action)` will consume these
action strings directly.
"""

from collections import deque

from src.config.settings import DWELL_STEPS


# ── THE LOOKUP TABLE ──────────────────────────────────────────────────────────
# Maps a smoothed BCI decision string → robotic arm command string.
# Adding a new class (e.g. FEET → MOVE_DOWN) only requires editing here.
ACTION_LOOKUP: dict[str, str] = {
    "LEFT":  "MOVE_LEFT",
    "RIGHT": "MOVE_RIGHT",
    "STOP":  "HOLD",
}
# ─────────────────────────────────────────────────────────────────────────────


class ControlPolicy:
    """
    Converts a smoothed BCI decision → verified robot action.

    Parameters
    ----------
    dwell_steps : int — number of consecutive identical decisions
                        required before issuing the action (default 3)
    """

    def __init__(self, dwell_steps: int = DWELL_STEPS):
        self.dwell_steps     = dwell_steps
        self._recent         = deque(maxlen=dwell_steps)
        self._current_action = "HOLD"

    def get_action(self, decision: str) -> str:
        """
        Parameters
        ----------
        decision : str — output of DecisionSmoother.update()

        Returns
        -------
        str — one of the values in ACTION_LOOKUP (e.g. "MOVE_LEFT")

        How the lookup table is used:
          Once dwell is satisfied, ACTION_LOOKUP[decision] is returned.
          If dwell is not yet satisfied, the previous stable action is kept.
        """
        self._recent.append(decision)

        # Dwell check: all recent decisions must be the same
        if (
            len(self._recent) == self.dwell_steps
            and all(d == decision for d in self._recent)
        ):
            # ── LOOKUP TABLE APPLIED HERE ─────────────────────────────────────
            self._current_action = ACTION_LOOKUP.get(decision, "HOLD")

        return self._current_action

    @property
    def current_action(self) -> str:
        return self._current_action

    def reset(self):
        self._recent.clear()
        self._current_action = "HOLD"

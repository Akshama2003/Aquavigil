"""Turn a ranked list of suspect pipes into an inspection work order.

The work order itself is deterministic, so it works offline and in tests.
`narrate()` optionally asks a Strands agent (Amazon Bedrock) to rewrite it as a
short plain-language message for a crew supervisor. If Strands or Bedrock
credentials are not available it returns None and the deterministic text is used.
"""

from .network import Network
from .simulate import STEP_HOURS, STEPS

# PLACEHOLDER ASSUMPTION: cubic metres lost per day per metre of fitted leak size.
# Calibrate this against real flow data from a utility before quoting loss figures.
LOSS_M3_PER_DAY_PER_UNIT = 8.0


def _clock(step_of_day: int) -> str:
    minutes = int(step_of_day * STEP_HOURS * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build_work_order(net: Network, alarm_step_of_day: int, ranking: list, k: int = 5):
    suspects = ranking[:k]
    top = suspects[0]
    est_loss = top["size"] * LOSS_M3_PER_DAY_PER_UNIT
    lines = [
        "# Leak inspection work order",
        "",
        f"Alarm raised at {_clock(alarm_step_of_day % STEPS)} (network time).",
        f"Estimated loss if the top suspect is correct: ~{est_loss:.0f} m3/day "
        "(indicative only, uncalibrated).",
        "",
        "Inspect in this order and stop when the leak is found:",
    ]
    for i, s in enumerate(suspects, 1):
        lines.append(f"{i}. {net.pipe_label(s['pipe'])} - fitted size {s['size']:.2f}")
    lines += [
        "",
        "Note: this is a ranked shortlist from pressure data, not a confirmed location.",
    ]
    return {
        "markdown": "\n".join(lines),
        "suspects": [
            {"pipe": s["pipe"], "label": net.pipe_label(s["pipe"]), "size": s["size"]}
            for s in suspects
        ],
        "est_loss_m3_per_day": est_loss,
    }


def narrate(markdown: str):
    """Optional: plain-language rewrite via a Strands agent. Returns None if unavailable."""
    try:
        from strands import Agent

        agent = Agent(
            system_prompt=(
                "You write short, clear instructions for water-utility field crews. "
                "Rewrite the work order in under 80 words. Do not add facts."
            )
        )
        return str(agent(markdown))
    except Exception:
        return None

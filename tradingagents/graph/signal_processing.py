import re


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    _DECISIONS = ["NOT BUY", "SELL", "BUY", "HOLD"]

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (NOT BUY, BUY, SELL, or HOLD)
        """
        text = full_signal.upper()
        for decision in self._DECISIONS:
            if re.search(rf'\b{re.escape(decision)}\b', text):
                return decision
        return "HOLD"
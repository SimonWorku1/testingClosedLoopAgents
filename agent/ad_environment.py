import random


class AdEnvironment:
    def __init__(self):
        self.total_budget = 1000.0
        self._true_cpa = {
            "search": 15.0,
            "social": 22.0,
            "video": 35.0,
            "display": 8.0,
        }

    def spend_daily_budget(self, allocations: dict) -> dict:
        """
        Accepts a dict of allocations, e.g., {"search": 50, "social": 50, ...}
        Returns spend totals, sign-ups per channel, and remaining budget.
        """
        total_allocated = sum(allocations.values())
        if total_allocated > self.total_budget + 0.01:
            raise ValueError(
                f"Allocation ${total_allocated:.2f} exceeds remaining budget ${self.total_budget:.2f}"
            )

        self.total_budget -= total_allocated
        total_sign_ups = 0
        channel_results = {}

        for channel, spend in allocations.items():
            if spend <= 0:
                channel_results[channel] = {"spend": 0, "sign_ups": 0, "effective_cpa": 0.0}
                continue

            saturation_penalty = 1 + (spend / 200.0)
            current_cpa = self._true_cpa[channel] * saturation_penalty
            current_cpa *= random.uniform(0.9, 1.1)

            channel_sign_ups = int(spend / current_cpa)
            total_sign_ups += channel_sign_ups

            channel_results[channel] = {
                "spend": spend,
                "sign_ups": channel_sign_ups,
                "effective_cpa": round(current_cpa, 2),
            }

        return {
            "spend": total_allocated,
            "sign_ups": total_sign_ups,
            "remaining_budget": self.total_budget,
            "channel_results": channel_results,
        }

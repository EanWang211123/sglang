"""Batch-size aware speculative decoding parameters.

Maps the current decode batch size directly to a fixed speculative_num_steps,
allowing operators to tune the speculation budget per batch-size regime.

Unlike the EMA-based adaptive strategy, this strategy is deterministic and
requires no warm-up: the step count is resolved instantly from a lookup table
the moment the batch size is known.
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_batch_size_aware_config(path: str | None) -> dict:
    """Load batch-size aware speculative config from a JSON file.

    Expected JSON format::

        {
            "batch_size_to_steps": {
                "1": 7,
                "2": 7,
                "4": 3,
                "8": 1
            }
        }

    Keys must be stringified integers (JSON object keys are always strings).
    Batch sizes that are absent from the table will use the server's default
    ``speculative_num_steps``.

    Returns an empty dict when *path* is ``None``.
    """
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            "batch-size-aware speculative config must be a JSON object, "
            f"got {type(cfg).__name__}"
        )
    if "batch_size_to_steps" not in cfg:
        raise ValueError(
            "batch-size-aware speculative config must contain "
            "a 'batch_size_to_steps' key"
        )
    mapping = cfg["batch_size_to_steps"]
    if not isinstance(mapping, dict):
        raise ValueError(
            "'batch_size_to_steps' must be a JSON object mapping "
            "batch-size strings to integer step counts"
        )
    return cfg


class BatchSizeAwareSpecParams:
    """Resolves speculative_num_steps from the current batch size.

    The mapping is built as follows:
    - For each batch size present in the user-supplied ``batch_size_to_steps``
      table the corresponding step count is used.
    - For all other batch sizes the server-level ``default_steps`` is used.

    Only the *unique* step values that appear in the final mapping are exposed
    via ``candidate_steps``; these are the step configurations that
    ``AdaptiveController`` will build ``SpecRuntimeState`` objects for.

    ``current_steps`` tracks the last resolved step value so that
    ``AdaptiveController`` can detect transitions and swap runtime states.
    """

    def __init__(
        self,
        default_steps: int,
        config: dict | None = None,
    ):
        cfg = config or {}
        raw_mapping: dict = cfg.get("batch_size_to_steps", {})

        # Validate and convert string keys to int
        self._user_mapping: dict[int, int] = {}
        for k, v in raw_mapping.items():
            try:
                bs = int(k)
                steps = int(v)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"batch_size_to_steps keys and values must be integers, "
                    f"got key={k!r}, value={v!r}"
                ) from e
            if bs <= 0:
                raise ValueError(
                    f"batch_size_to_steps keys must be positive integers, got {bs}"
                )
            if steps <= 0:
                raise ValueError(
                    f"batch_size_to_steps values must be positive integers, "
                    f"got steps={steps} for batch_size={bs}"
                )
            self._user_mapping[bs] = steps

        self._default_steps = default_steps

        # candidate_steps: union of all user-specified steps plus the default
        self.candidate_steps: list[int] = sorted(
            set(self._user_mapping.values()) | {default_steps}
        )

        # Start at the default; first real batch update may switch immediately
        self.current_steps = default_steps

        logger.info(
            "BatchSizeAwareSpecParams initialized: "
            f"default_steps={default_steps}, "
            f"candidate_steps={self.candidate_steps}, "
            f"user_overrides={self._user_mapping}"
        )

    def batch_sizes_for_step(self, step: int, all_batch_sizes: list[int]) -> list[int]:
        """Return the subset of *all_batch_sizes* that are routed to *step*."""
        return [
            bs
            for bs in all_batch_sizes
            if self._user_mapping.get(bs, self._default_steps) == step
        ]

    def update(self, batch_size: int) -> bool:
        """Resolve steps for *batch_size*. Returns True if the step changed.

        Batch sizes not present in the user mapping fall back to
        ``default_steps``.
        """
        target = self._user_mapping.get(batch_size, self._default_steps)
        if target != self.current_steps:
            old = self.current_steps
            self.current_steps = target
            logger.debug(
                f"BatchSizeAware spec: steps {old} -> {target} "
                f"(batch_size={batch_size})"
            )
            return True
        return False

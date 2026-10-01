from dataclasses import dataclass

@dataclass(frozen=True)
class SymCorePolicy:
    min_sequence_length: int = 1536
    min_expected_ratio: float = 4.0

    def should_attempt(self, sequence_length: int, expected_ratio: float | None = None) -> bool:
        if sequence_length < self.min_sequence_length:
            return False
        if expected_ratio is not None and expected_ratio < self.min_expected_ratio:
            return False
        return True

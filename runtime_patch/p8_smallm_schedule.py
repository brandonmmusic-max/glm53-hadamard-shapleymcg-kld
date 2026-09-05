"""CPU-visible geometry for the developmental, exact-contract P8 M1 arm."""
from dataclasses import dataclass


@dataclass(frozen=True)
class P8SmallMGeometry:
    tokens: int = 1
    experts: int = 288
    hidden: int = 4096
    intermediate: int = 512
    topk: int = 8

    def __post_init__(self):
        if (self.tokens, self.experts, self.hidden, self.intermediate, self.topk) != (
            1, 288, 4096, 512, 8
        ):
            raise ValueError("small-M candidate is frozen to GLM TP4 M1")

    @property
    def physical_tiles(self):
        return self.topk

    @property
    def fc1_tasks(self):
        return self.topk * (self.intermediate // 128)

    @property
    def fc2_tasks(self):
        return self.topk * (self.hidden // 256)

    def fc1_owner(self, slot):
        if not 0 <= slot < self.fc1_tasks:
            raise ValueError("FC1 task outside capacity")
        return divmod(slot, self.intermediate // 128)

    def fc2_owner(self, slot):
        if not 0 <= slot < self.fc2_tasks:
            raise ValueError("FC2 task outside capacity")
        return divmod(slot, self.hidden // 256)


def use_small_m(enabled: bool, tokens: int) -> bool:
    return bool(enabled and tokens == 1)

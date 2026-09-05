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


@dataclass(frozen=True)
class P8ScratchRegion:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class P8ScratchLayout:
    regions: tuple[P8ScratchRegion, ...]
    nbytes: int


def p8_small_m_scratch_layout() -> P8ScratchLayout:
    """Exact M1 buffer extents, each starting at a 16-byte-aligned offset.

    The route-output allocation is intentionally excluded. Padding between
    regions is also zeroed by the caller's single uint8 arena initialization.
    """
    geometry = P8SmallMGeometry()
    rows = geometry.physical_tiles * 16
    max_tasks = geometry.physical_tiles * (geometry.intermediate // 128)
    specs = [
        ("packed_a", "uint8", (rows * geometry.hidden,)),
        ("scale_flat", "uint8",
         ((geometry.experts + geometry.topk + 1) * 16 * (geometry.hidden // 8),)),
        ("intermediate_u32", "int32",
         (rows * (geometry.intermediate + geometry.intermediate // 32) // 4,)),
    ]
    specs.extend((name, "int32", (1,)) for name in (
        "barrier_count", "barrier_epoch", "pair_head", "producers_done",
        "all_published", "task_head", "task_tail",
    ))
    specs.extend((name, "int32", (max_tasks,)) for name in (
        "task_ready", "task_expert", "task_m_tile", "task_slice_begin",
        "task_slice_count", "task_valid_rows",
    ))
    specs.extend([
        ("tile_write_count", "int32", (geometry.physical_tiles,)),
        ("row_counts", "int32", (geometry.experts,)),
        ("expert_write_rows", "int32", (geometry.experts,)),
        ("expert_tile_base", "int32", (geometry.experts + 1,)),
        ("token_map", "int32", (rows,)),
        ("token_weights", "float32", (rows,)),
        ("output", "bfloat16", (1, geometry.hidden)),
    ])
    sizes = {"uint8": 1, "int32": 4, "float32": 4, "bfloat16": 2}
    regions = []
    cursor = 0
    for name, dtype, shape in specs:
        cursor = (cursor + 15) // 16 * 16
        elements = 1
        for extent in shape:
            elements *= extent
        nbytes = elements * sizes[dtype]
        regions.append(P8ScratchRegion(name, dtype, shape, cursor, nbytes))
        cursor += nbytes
    return P8ScratchLayout(tuple(regions), (cursor + 15) // 16 * 16)

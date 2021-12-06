from dataclasses import dataclass
from typing import Dict, List

@dataclass
class FuncPerf:
    name: str
    runtime: float
    threads: float
    peak_usage: float
    stack_usage: float
    num_allocations: float
    avg_size_allocations: float


@dataclass
class ProgramPerf:
    runs: int
    avg_threads: float
    heap_allocations: float
    peak_heap_usage: float
    func_perf_counters: List[FuncPerf]


@dataclass
class SampleDict:
    sample_name: str
    actual_runtime: float
    predicted_runtime: float
    compiled_features: Dict[str, Dict[str, float]]
    perf: ProgramPerf

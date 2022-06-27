from dataclasses import dataclass
from typing import Counter, Dict, List

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
class CounterDict:
    store_counters: Dict[str, int]
    scalar_load_counters: Dict[str, int]
    vector_load_counters: Dict[str, int]


@dataclass
class SampleDict:
    sample_name: str
    actual_runtime: float
    predicted_runtime: float
    compiled_features: Dict[str, Dict[str, float]]
    perf: ProgramPerf
    trace_counters: CounterDict
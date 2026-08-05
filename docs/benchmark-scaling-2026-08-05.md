# Training Scaling Benchmark — 2026-08-05

## Result

On the tested 4-core/8-thread mobile CPU, neural-c scales approximately
linearly with dataset size after fixed startup costs, but its current training
executor does not scale linearly with worker count. Eight workers produced the
lowest median wall time, with speedups between 1.22x and 1.42x rather than 8x.

This is an end-to-end training benchmark: project parsing, dataset loading,
model initialization, two training epochs, and final atomic weights persistence
are all included.

## Method

The test follows a conventional strong-scaling protocol: each dataset and model
remain fixed while the worker count changes through 1, 2, 4, and 8. Each point
uses one unrecorded warm-up followed by five fresh-process measurements. The
reported wall time is the median; the range shows the minimum and maximum.

- Revision: `4d4a80c` (`Add optimizers and convergence control`)
- Build: native x86-64, GCC 13.3.0, production `-O2` flags
- Host: Intel Core i7-1365U, 4 physical cores, 8 logical CPUs, 15 GiB RAM
- Runtime: Linux 6.18 under WSL2
- Affinity: logical CPUs 0-7
- Initial system load: 0.39, 0.17, 0.18
- Runs: sequential, one benchmark process at a time

The deterministic synthetic classification workload avoids network, storage,
and third-party preprocessing dependencies:

- 32 finite numeric inputs and 8 one-hot classes;
- dense `32 -> 64 relu -> 32 relu -> 8 softmax` model;
- categorical cross-entropy, Adam, learning rate 0.001;
- full-dataset batch, no shuffle, no periodic checkpoint;
- two epochs per measured run;
- sample inputs generated from fixed sine/cosine functions and class `i % 8`.

The three sizes deliberately keep the total benchmark bounded:

| Name | Samples | Dataset file |
| --- | ---: | ---: |
| Small | 1,000 | 0.38 MiB |
| Medium | 10,000 | 3.88 MiB |
| Extended | 50,000 | 19.43 MiB |

## Timing results

Throughput is expressed as sample-epochs per second: samples multiplied by two
epochs, divided by median wall time. Speedup is `T1 / Tp`; efficiency is
`speedup / workers`.

| Dataset | Workers | Median | Range | Throughput | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 1 | 0.131 s | 0.129–0.160 s | 15,287 | 1.000x | 100.0% |
| Small | 2 | 0.156 s | 0.153–0.158 s | 12,783 | 0.836x | 41.8% |
| Small | 4 | 0.122 s | 0.119–0.126 s | 16,358 | 1.070x | 26.8% |
| Small | 8 | 0.107 s | 0.104–0.114 s | 18,620 | 1.218x | 15.2% |
| Medium | 1 | 1.778 s | 1.603–1.881 s | 11,248 | 1.000x | 100.0% |
| Medium | 2 | 1.709 s | 1.466–1.827 s | 11,704 | 1.041x | 52.0% |
| Medium | 4 | 1.418 s | 1.254–1.498 s | 14,109 | 1.254x | 31.4% |
| Medium | 8 | 1.257 s | 1.230–1.309 s | 15,917 | 1.415x | 17.7% |
| Extended | 1 | 8.698 s | 7.636–9.250 s | 11,497 | 1.000x | 100.0% |
| Extended | 2 | 8.583 s | 8.414–8.784 s | 11,651 | 1.013x | 50.7% |
| Extended | 4 | 7.298 s | 7.117–7.543 s | 13,702 | 1.192x | 29.8% |
| Extended | 8 | 6.586 s | 6.422–6.653 s | 15,184 | 1.321x | 16.5% |

Dataset growth is close to linear once the workload is large enough. At one
worker, increasing from 10,000 to 50,000 samples multiplies work by 5 and
median time by 4.89. At eight workers, the same transition multiplies time by
5.24.

## Memory and CPU observations

Peak RSS and average process CPU were sampled once per configuration after the
timing experiment. They are diagnostic observations, not five-run medians.

| Dataset | 1 worker | 2 workers | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| Small RSS | 3.05 MiB | 3.23 MiB | 3.18 MiB | 3.24 MiB |
| Medium RSS | 5.83 MiB | 5.80 MiB | 5.97 MiB | 6.24 MiB |
| Extended RSS | 18.03 MiB | 18.21 MiB | 18.18 MiB | 18.41 MiB |
| Small CPU | 86% | 101% | 120% | 130% |
| Medium CPU | 89% | 103% | 120% | 138% |
| Extended CPU | 88% | 105% | 121% | 142% |

Memory grows primarily with sample count and barely changes with worker count
for this small model. The 8-worker process averaged only 1.30–1.42 logical CPUs
of utilization, which is consistent with the limited wall-time speedup.

## Interpretation

- Dataset-size scalability is healthy: time and memory are approximately
  linear rather than super-linear for the medium and extended workloads.
- Worker scalability is currently modest. Two workers may be neutral or slower;
  four and eight improve latency, but with low parallel efficiency.
- For this host, use one worker for very small jobs, four workers as a balanced
  default for non-trivial jobs, and eight only when the extra 10–12% reduction
  over four workers matters more than CPU efficiency.
- The most likely limiting factor is the correctness-first executor contract:
  workers synchronize after every bounded wave, then the coordinator performs
  deterministic ordered accumulation and the exclusive update. This is an
  inference from the implementation and CPU-utilization measurements, not a
  sampling-profiler result.

A future performance checkpoint should profile mutex/condition waits and
coordinator accumulation, then investigate processing multiple samples per
dispatch while preserving the documented deterministic reduction order and
bounded memory contract.

## Limitations

Absolute times are specific to a mobile CPU under WSL2 with uncontrolled turbo
frequency. The medians are suitable for comparing configurations in this run,
not for comparing different machines. This benchmark covers dense-model
training; prediction, evaluation, native ppc64le hardware, sparse workloads,
and wider/deeper models require separate measurements.

from typing import Callable
import multiprocessing as mp
from enum import Enum


class Signals(Enum):
    SAMPLE_CONSUMED = 1


# --- FIX 1: Identity Decorator ---
# Returns the class exactly as it is, preserving its identity so
# Python's multiprocessing pickle can serialize it without crashing.
def StepConverter(cls):
    return cls


class BaseProcess:
    def __init__(self, num_processes=1, size_queue=1000):
        self.num_processes = num_processes
        self.size_queue = size_queue


class Pipeline:
    def __init__(self, steps, total_samples=None):
        self.steps = steps
        self.total_samples = total_samples

    def _worker(self, step_func: Callable, input_queue: mp.Queue | None, output_queue: mp.Queue) -> None:
        if input_queue is None:
            # Stage 1: Producer
            for item in step_func():
                output_queue.put(item)
            # Send the initial STOP signal when links run out
            output_queue.put("STOP")
        else:
            # Stage 2+: Consumers/Transformers
            def queue_generator():
                while True:
                    val = input_queue.get()
                    if val == "STOP":
                        # --- FIX 2: Deadlock Prevention ---
                        # Put the STOP signal back on the queue so the other
                        # 15 sibling processes can also read it and shut down.
                        input_queue.put("STOP")
                        break
                    yield val

            for item in step_func(queue_generator()):
                if output_queue and item != Signals.SAMPLE_CONSUMED:
                    output_queue.put(item)

            # Pass the STOP signal down the pipeline
            if output_queue:
                output_queue.put("STOP")

    def run(self, debug_inspect_queue_sizes: bool = False) -> None:
        queues: list[mp.Queue] = []
        processes: list[mp.Process] = []

        for i in range(len(self.steps) - 1):
            size = getattr(self.steps[i + 1], "size_queue", 1000)
            queues.append(mp.Queue(maxsize=size))

        for i, step in enumerate(self.steps):
            in_q = queues[i - 1] if i > 0 else None
            out_q = queues[i] if i < len(self.steps) - 1 else None

            for _ in range(step.num_processes):
                p = mp.Process(target=self._worker, args=(step.__call__, in_q, out_q))
                p.start()
                processes.append(p)

        for p in processes:
            print(f"Joining process {p.pid}")
            p.join()

        for p in processes:
            print(f"Terminating process {p.pid}")
            if p.is_alive():
                p.terminate()
            p.join(timeout=1)
            p.close()

        for q in queues:
            q.close()
            q.join_thread()

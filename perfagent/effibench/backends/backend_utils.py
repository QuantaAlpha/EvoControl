import os
import queue
import random
import logging
import requests
import platform
import subprocess
import threading
import multiprocessing
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Optional, Protocol, Type, TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from ..utils import EFFIBENCH_REGISTRY

# -------------------------------------------------------------------
# CPU Topology Detection
# -------------------------------------------------------------------


@cache
def get_cpu_topology() -> dict[int, list[int]]:
    """Return a dict of { physical_core: [logical_cpus] }."""
    topology = {}
    try:
        if platform.system() == "Linux":
            out = subprocess.check_output("lscpu -p=CPU,CORE", shell=True).decode()
            for line in out.splitlines():
                if line and not line.startswith("#"):
                    logical_id, physical_id = map(int, line.split(","))
                    topology.setdefault(physical_id, []).append(logical_id)
        elif platform.system() == "Darwin":
            cmd = "sysctl -n hw.physicalcpu hw.logicalcpu"
            physical, logical = map(int, subprocess.check_output(cmd, shell=True).decode().split())
            if logical > physical:
                tpc = logical // physical
                for p in range(physical):
                    topology[p] = [p * tpc + t for t in range(tpc)]
        if not topology:
            count = multiprocessing.cpu_count()
            for i in range(count):
                topology[i] = [i]
    except Exception as e:
        logging.warning(f"Cannot detect CPU topology: {e}")
        count = multiprocessing.cpu_count()
        for i in range(count):
            topology[i] = [i]
    return topology


def get_num_physical_cores() -> int:
    return len(get_cpu_topology())


def set_cpu_affinity(pid: int, cpu_list: list[int]):
    """
    Set CPU affinity for a process across platforms.

    Args:
        pid: Process ID
        cpu_list: List of CPU cores to bind to
    """
    system = platform.system()

    if system == "Linux":
        # Linux implementation using taskset
        cpu_str = ",".join(map(str, cpu_list))
        cmd = f"taskset -pc {cpu_str} {pid}"
        try:
            subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.warning(f"Failed to set CPU affinity for {pid} to {cpu_str}")
    else:
        logging.warning(f"CPU affinity currently not implemented for {system}")


def bind_main_process_to_last_core():
    """
    Bind the main process to the last physical core.
    This is useful to separate the main process from worker processes.
    """
    topology = get_cpu_topology()
    if not topology:
        return
    # Get the last physical core
    last_physical_core = max(topology.keys())
    # Get the logical CPUs for the last physical core
    last_core_logical_cpus = topology[last_physical_core]
    # Bind the current process to the last core
    current_pid = os.getpid()
    set_cpu_affinity(current_pid, last_core_logical_cpus)


# -------------------------------------------------------------------
# Efficiency Calculation
# -------------------------------------------------------------------


def calculate_efficiency_integral(times: list[int], mems: list[int]) -> float:
    """
    Calculate the area under the memory-time curve (efficiency integral).

    Args:
        times: List of timestamps in nanoseconds
        mems: List of memory values in KB

    Returns:
        float: The integral value in MB·s
    """
    if len(times) != len(mems):
        return 0.0

    # Need at least two points to calculate area
    if len(times) < 2:
        return 0.0

    # Convert to seconds and MB for consistent units
    times_sec = np.array(times) / 1000000000  # nanoseconds to seconds
    mems_mb = np.array(mems) / 1000  # KB to MB

    # Calculate area using trapezoidal rule
    area = np.trapz(mems_mb, times_sec)
    return area


# -------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------


class CodeExecutionRequest(BaseModel):
    code: str
    language: str
    libraries: Optional[list[str]] = None
    stdin: Optional[str] = None
    time_limit: Optional[int] = None
    memory_limit: Optional[float] = None


class BatchSubmissionRequest(BaseModel):
    requests: list[CodeExecutionRequest]


class SubmissionResponse(BaseModel):
    submission_id: int


class BatchSubmissionResponse(BaseModel):
    submission_ids: list[int]


class BatchGetSubmissionRequest(BaseModel):
    submission_ids: list[int]


class CodeExecutionResponse(BaseModel):
    submission_id: int
    status: str = Field(..., description="waiting|processing|done|timeout|error|cancelled|oom")
    text: Optional[str] = None
    exit_code: Optional[int] = None
    runtime: Optional[float] = None
    memory: Optional[float] = None
    integral: Optional[float] = None


class BatchCodeExecutionResponse(BaseModel):
    results: list[CodeExecutionResponse]


class CancelResponse(BaseModel):
    submission_id: int
    status: str


# -------------------------------------------------------------------
# Common Utilities for Managers
# -------------------------------------------------------------------


def parse_memory_profile(profile_text: str, skip_header: bool = True) -> tuple[list[int], list[int]]:
    """Parse lines of 'timestamp memory' into two lists."""
    times, mems = [], []
    base_time = None
    for line in profile_text.splitlines():
        if skip_header:
            skip_header = False
            continue
        try:
            t, m = map(int, line.split())
            if base_time is None:
                base_time = t
            times.append(t - base_time)
            mems.append(m)
        except:
            pass
    return times, mems


# -------------------------------------------------------------------
# Submission Record
# -------------------------------------------------------------------


@dataclass
class SubmissionRecord:
    """Represents a code execution submission with its own lock for concurrency control."""

    id: int
    request: "CodeExecutionRequest"
    status: str = "waiting"
    exit_code: Optional[int] = None
    text: Optional[str] = None
    runtime: Optional[float] = None
    memory: Optional[float] = None
    integral: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert submission record to a dictionary (excluding the lock)."""
        return {
            "submission_id": self.id,
            "status": self.status,
            "exit_code": self.exit_code,
            "text": self.text,
            "runtime": self.runtime,
            "memory": self.memory,
            "integral": self.integral,
        }


# -------------------------------------------------------------------
# Session Protocol
# -------------------------------------------------------------------


class SandboxSessionProtocol(Protocol):
    """Protocol defining the interface required for sandbox sessions."""

    def open(self, skip_setup: bool = False) -> None:
        """Open the session and prepare resources."""
        ...

    def close(self) -> None:
        """Close the session and clean up resources."""
        ...

    def run(
        self,
        code: str,
        libraries: Optional[list] = None,
        stdin: Optional[str] = None,
        time_limit: Optional[float] = None,
        memory_limit: Optional[float] = None,
        return_statistics: bool = True,
    ) -> Any:
        """
        Execute code in the session and return results.

        Args:
            code: Code to run
            libraries: List of libraries to install
            stdin: Standard input to pass to the program
            time_limit: Maximum execution time in seconds (None or 0 = no time limit)
            memory_limit: Memory limit in MB (None or 0 = no memory limit)
            return_statistics: If True, calculate and return execution statistics (runtime, memory, integral)
        """
        ...

    def cat_profile(self) -> Optional[str]:
        """Return the profiling information for the last execution."""
        ...


# -------------------------------------------------------------------
# Base Execution Manager
# -------------------------------------------------------------------

T = TypeVar("T", bound=SandboxSessionProtocol)


class BaseExecutionManager(ABC):
    """
    submissions[id] = {
        "id": int,
        "request": CodeExecutionRequest,
        "status": str,  # waiting|processing|done|cancelled|error|timeout|oom
        "exit_code": Optional[int],
        "text": Optional[str],
        "runtime": Optional[float],
        "memory": Optional[float],
        "integral": Optional[float],
    }
    """

    def __init__(
        self,
        session_class: Type[T],
        num_workers: Optional[int] = None,
        allowed_languages: Optional[list[str]] = None,
        skip_setup: bool = False,
    ):
        if num_workers is None:
            num_workers = get_num_physical_cores() - 1
        self.num_workers = num_workers
        self.session_class = session_class
        # Allowed languages (registry keys). If None, allow all from registry
        self.allowed_languages = (
            list(EFFIBENCH_REGISTRY.keys()) if allowed_languages is None else list(allowed_languages)
        )

        # Request queue
        self.req_queue = queue.Queue()

        # Submission tracking
        self.submission_id = 0
        self.submissions: dict[int, SubmissionRecord] = {}
        self.global_lock = threading.Lock()  # Used only for dict operations (add/remove)

        # Session tracking
        self.sessions: dict[tuple[int, str], T] = {}
        self.sessions_lock = threading.Lock()

        if skip_setup:
            logging.info("Skipping setup.")
        else:
            self.setup()

    def setup(self):
        """Initialize session environments for all registered languages (filtered by allowed_languages)."""
        for lang, config in EFFIBENCH_REGISTRY.items():
            if lang not in self.allowed_languages:
                continue
            self._create_initial_session(lang, config)

    @abstractmethod
    def _create_initial_session(self, lang: str, config: dict) -> None:
        """Create an initial session for setup purposes. Implemented by subclasses."""
        pass

    @abstractmethod
    def get_session(self, worker_id: int, language: str) -> T:
        """Get or create a session for the worker. Implemented by subclasses."""
        pass

    def start_workers(self):
        """Start worker threads for parallel code execution."""
        for i in range(self.num_workers):
            th = threading.Thread(target=self.worker_loop, args=(i,), daemon=True)
            th.start()
            logging.info(f"Worker {i} started.")

    def stop_workers(self):
        """Stop all worker threads and clean up sessions."""
        with self.sessions_lock:
            for session in self.sessions.values():
                session.close()

    def _process_single_submission(self, worker_id: int, sid: int):
        """Processes a single submission ID."""
        # Get the record reference without locking
        record = self.submissions.get(sid)
        if record is None:
            logging.error(f"[Worker {worker_id}] Submission {sid} not found.")
            return  # Skip if submission vanished somehow

        # Use the record's lock for state checking and modification
        with record.lock:
            if record.status == "cancelled":
                logging.debug(f"[Worker {worker_id}] Skipping cancelled submission {sid}")
                return  # Skip if cancelled before processing started

            record.status = "processing"

        # Execute without holding the lock
        try:
            self.execute_in_sandbox(worker_id, record)
        except Exception as e:
            # Only lock when updating the record
            if sid in self.submissions:  # Check if record still exists
                with record.lock:
                    record.status = "error"
                    record.exit_code = 1
                    record.text = f"{e.__class__.__name__}: {str(e)}"

    def worker_loop(self, worker_id: int):
        """Main worker loop. Handles single submissions (int) or batches (list[int])."""
        while True:
            item = self.req_queue.get()
            logging.info(f"[Worker {worker_id}] Got item. Queue size: {self.req_queue.qsize()}")
            try:
                if isinstance(item, int):
                    # Process single submission ID
                    logging.debug(f"[Worker {worker_id}] Received single submission {item}")
                    self._process_single_submission(worker_id, item)
                elif isinstance(item, list):
                    # Process batch of submission IDs sequentially
                    logging.debug(f"[Worker {worker_id}] Received batch of {len(item)} submissions: {item}")
                    for sid in item:
                        # Check if cancelled globally before processing each item in batch
                        # No lock needed here as _process_single_submission handles the check+update atomically
                        self._process_single_submission(worker_id, sid)
                    logging.debug(f"[Worker {worker_id}] Finished batch {item}")
                else:
                    logging.error(f"[Worker {worker_id}] Received unexpected item type from queue: {type(item)}")
            except Exception as e:
                # Catch broad exceptions during the loop logic itself (e.g., queue issues)
                logging.error(
                    f"[Worker {worker_id}] Unhandled error in worker loop: {e}",
                    exc_info=True,
                )
            finally:
                # Mark task done regardless of item type or processing errors inside the loop
                self.req_queue.task_done()

    def _add_submission_entry(self, req: CodeExecutionRequest) -> int:
        """Adds a single submission entry and returns the ID. Assumes global_lock is held."""
        sid = self.submission_id
        self.submission_id += 1
        self.submissions[sid] = SubmissionRecord(id=sid, request=req)
        return sid

    def submit_code(self, req: CodeExecutionRequest) -> int:
        """Submit code for execution and return a submission ID."""
        if not req.code.strip():
            raise HTTPException(status_code=400, detail="No code provided.")
        if not req.language.strip():
            raise HTTPException(status_code=400, detail="No language specified.")

        # Validate allowed languages (accept both registry keys and mapped sandbox langs)
        allowed_set = set(self.allowed_languages)
        allowed_set.update(
            {
                cfg.get("llm_sandbox_lang")
                for lang_key, cfg in EFFIBENCH_REGISTRY.items()
                if lang_key in self.allowed_languages
            }
        )
        if req.language not in allowed_set:
            raise HTTPException(
                status_code=400,
                detail=f"Language '{req.language}' not allowed. Allowed: {sorted(list(allowed_set))}.",
            )

        with self.global_lock:
            sid = self._add_submission_entry(req)

        self.req_queue.put(sid)
        return sid

    def submit_batch(self, batch_req: BatchSubmissionRequest) -> list[int]:
        """Submit a batch of code execution requests and return their IDs."""
        if not batch_req.requests:
            raise HTTPException(status_code=400, detail="No requests provided in the batch.")

        # Validate all requests before acquiring the lock
        for req in batch_req.requests:
            if not req.code.strip():
                raise HTTPException(status_code=400, detail=f"Request with empty code found in batch.")
            if not req.language.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Request with unspecified language found in batch.",
                )
            allowed_set = set(self.allowed_languages)
            allowed_set.update(
                {
                    cfg.get("llm_sandbox_lang")
                    for lang_key, cfg in EFFIBENCH_REGISTRY.items()
                    if lang_key in self.allowed_languages
                }
            )
            if req.language not in allowed_set:
                raise HTTPException(
                    status_code=400,
                    detail=f"Language '{req.language}' not allowed in batch. Allowed: {sorted(list(allowed_set))}.",
                )

        submission_ids = []
        # Use smaller critical section to reduce lock contention
        with self.global_lock:
            for req in batch_req.requests:
                sid = self._add_submission_entry(req)
                submission_ids.append(sid)

        self.req_queue.put(submission_ids)

        return submission_ids

    def fetch_batch_results(self, submission_ids: list[int]) -> list[dict[str, Any]]:
        """Fetch the results for a batch of submission IDs."""
        results = []

        # No global lock needed - use per-record locking
        for sid in submission_ids:
            record = self.submissions.get(sid)
            if record is not None:
                with record.lock:
                    # Make a copy while holding record lock
                    results.append(record.to_dict())

        return results

    def fetch_result(self, submission_id: int) -> dict[str, Any]:
        """Fetch the result of a submission by ID."""
        record = self.submissions.get(submission_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Submission not found.")

        with record.lock:
            # Get a copy of the submission data under per-record lock
            return record.to_dict()

    def cancel_submission(self, submission_id: int) -> dict[str, Any]:
        """Cancel a submission by ID."""
        record = self.submissions.get(submission_id)

        if record is None:
            raise HTTPException(status_code=404, detail="Submission not found.")

        with record.lock:
            # If waiting => mark cancelled
            if record.status == "waiting":
                record.status = "cancelled"
            # If processing => mark cancelled and capture the process to kill
            elif record.status == "processing":
                record.status = "cancelled"
                record.exit_code = 1
                record.text = ""

        return {"submission_id": submission_id, "status": "cancelled"}

    def execute_in_sandbox(self, worker_id: int, record: SubmissionRecord):
        """Execute code in the sandbox and process results."""
        req = record.request

        try:
            # Map registry language to sandbox language if applicable
            if req.language in EFFIBENCH_REGISTRY:
                sandbox_lang = EFFIBENCH_REGISTRY[req.language].get("llm_sandbox_lang", req.language)
            else:
                sandbox_lang = req.language
            session = self.get_session(worker_id, sandbox_lang)

            # Filter out libraries that are already installed
            libraries = []
            if req.libraries:  # Check if libraries is not None
                if req.language in EFFIBENCH_REGISTRY:  # Ensure language exists in registry
                    libraries = [l for l in req.libraries if l not in EFFIBENCH_REGISTRY[req.language]["packages"]]
                else:
                    libraries = req.libraries

            # Run with statistics to get metrics
            res = session.run(
                code=req.code,
                libraries=libraries,
                stdin=req.stdin,
                time_limit=req.time_limit,
                memory_limit=req.memory_limit,
                return_statistics=True,
            )

            # Process execution result
            text = res.text
            exit_code = res.exit_code
            status = "done"

            # Set appropriate status based on exit code
            if exit_code == 137:
                status = "oom"
                text = f"Memory limit exceeded ({req.memory_limit} MB)\n{text}"
            elif exit_code == 124:
                status = "timeout"
                text = f"Time limit exceeded ({req.time_limit} s)\n{text}"
            elif exit_code != 0:
                status = "error"

        except Exception as e:
            status = "error"
            exit_code = 1
            text = f"Execution error: {str(e)}"
            logging.error(f"Error executing code: {e}")
            res = None  # Ensure res is defined even in case of exception

        with record.lock:
            record.status = status
            record.exit_code = exit_code
            record.text = text if text is not None else ""

            # Set metrics from sandbox response if available
            if status == "done" and res is not None:
                # Runtime (convert from ns to seconds if needed)
                if hasattr(res, "runtime") and res.runtime is not None:
                    record.runtime = res.runtime
                else:
                    record.runtime = 0.0

                # Memory
                if hasattr(res, "memory") and res.memory is not None:
                    record.memory = res.memory
                else:
                    record.memory = 0.0

                # Integral
                if hasattr(res, "integral") and res.integral is not None:
                    record.integral = res.integral
                else:
                    record.integral = 0.0
            else:
                # Set defaults for failed executions
                record.runtime = 0.0
                record.memory = 0.0
                record.integral = 0.0


# -------------------------------------------------------------------
# FastAPI App Creation
# -------------------------------------------------------------------


def create_fastapi_app(manager_instance):
    """Create a FastAPI application with code execution endpoints."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Bind the main process to the last physical core to ensure it doesn't
        # compete with worker processes for CPU resources
        bind_main_process_to_last_core()

        # Start worker threads after binding the main process
        manager_instance.start_workers()
        yield
        manager_instance.stop_workers()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health", status_code=200)
    def health_check():
        """Health check endpoint for backend availability monitoring."""
        return {"status": "ok", "queue_size": manager_instance.req_queue.qsize()}

    @app.post("/submit", response_model=SubmissionResponse)
    def submit_code_endpoint(data: CodeExecutionRequest):
        sid = manager_instance.submit_code(data)
        return SubmissionResponse(submission_id=sid)

    @app.post("/submit_batch", response_model=BatchSubmissionResponse)
    def batch_submit_code_endpoint(data: BatchSubmissionRequest):
        sids = manager_instance.submit_batch(data)
        return BatchSubmissionResponse(submission_ids=sids)

    @app.get("/submission/{submission_id}", response_model=CodeExecutionResponse)
    def get_submission_endpoint(submission_id: int):
        info = manager_instance.fetch_result(submission_id)
        return CodeExecutionResponse(**info)

    @app.post("/submissions", response_model=BatchCodeExecutionResponse)
    def batch_get_submission_endpoint(data: BatchGetSubmissionRequest):
        results = manager_instance.fetch_batch_results(data.submission_ids)
        return BatchCodeExecutionResponse(results=[CodeExecutionResponse(**res) for res in results])

    @app.post("/cancel/{submission_id}", response_model=CancelResponse)
    def cancel_submission_endpoint(submission_id: int):
        res = manager_instance.cancel_submission(submission_id)
        return CancelResponse(**res)

    return app


# -------------------------------------------------------------------
# Backend Health Management
# -------------------------------------------------------------------


# Custom exception for backend availability errors
class BackendUnavailableError(Exception):
    """Raised when the sandbox backend is unavailable or unhealthy."""

    pass


class BackendManager:
    """Manages backend availability and selection with health checks."""

    def __init__(self, urls=None, health_check_interval=60):
        """
        Initialize backend manager with URLs and health check configuration.

        Args:
            urls: List of backend URLs (if None, loaded from environment)
            health_check_interval: Seconds between health checks for unavailable backends
        """
        self._urls = urls
        self._available_backends = {}  # Changed from set to dict: url -> queue_size
        self._unavailable_backends = set()
        self._last_check_times = {}
        self._health_check_interval = health_check_interval
        self._refresh_interval = 10
        self._last_refresh_time = 0
        self._lock = threading.RLock()
        self._random = random.Random()  # Thread-safe random instance
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize the backend list from environment variables or provided URLs."""
        with self._lock:
            if self._urls is None:
                raw_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
                self._urls = [url.strip() for url in raw_url.split(",")]

            # Check health of all backends
            for url in self._urls:
                is_healthy, queue_size = self._check_backend_health(url)
                if is_healthy and queue_size is not None:
                    self._available_backends[url] = queue_size
                else:
                    self._unavailable_backends.add(url)
                    self._last_check_times[url] = time.time()

    def _check_backend_health(self, url, timeout=5):
        """
        Check if a backend is healthy and return its queue size.

        Args:
            url: Backend URL to check
            timeout: Request timeout in seconds

        Returns:
            tuple[bool, int | None]: (True if backend is healthy, queue_size) or (False, None)
        """
        try:
            # Try to connect to health endpoint (or root if no dedicated health endpoint)
            health_url = f"{url}/health"
            response = requests.get(health_url, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return True, data.get("queue_size")
            return False, None
        except requests.RequestException:
            return False, None

    def mark_backend_unavailable(self, url):
        """Mark a backend as unavailable after a failed request."""
        with self._lock:
            # Remove from available backends if it's there
            if url in self._available_backends:
                del self._available_backends[url]

            # Add to unavailable backends
            if url not in self._unavailable_backends:  # Ensure not to add duplicates
                logging.warning(f"Backend {url} marked as unavailable")
                self._unavailable_backends.add(url)
                self._last_check_times[url] = time.time()

    def get_available_backend(self):
        """
        Get the least loaded available backend URL, updating queue sizes and rechecking unavailable backends.

        Returns:
            str: URL of an available backend or None if none are available
        """
        with self._lock:
            current_time = time.time()

            # Periodically refresh all available backends' queue sizes
            should_refresh = current_time - self._last_refresh_time >= self._refresh_interval
            if should_refresh:
                self._refresh_available_backends(current_time)
                self._last_refresh_time = current_time

            # Recheck unavailable backends
            self._recheck_unavailable_backends(current_time)

            # If we have available backends, select the one with the smallest queue size
            if self._available_backends:
                min_queue_size = min(self._available_backends.values())
                least_loaded_backends = [
                    url for url, q_size in self._available_backends.items() if q_size == min_queue_size
                ]

                if least_loaded_backends:
                    return self._random.choice(least_loaded_backends)  # Return a random one from the least loaded

            # If still no backends available after all checks
            logging.error("No available backends found after re-checks.")
            return None

    def _refresh_available_backends(self, current_time):
        """Refresh queue sizes for all available backends."""
        available_copy = list(self._available_backends.keys())
        for url in available_copy:
            is_healthy, queue_size = self._check_backend_health(url)
            if is_healthy and queue_size is not None:
                self._available_backends[url] = queue_size
                logging.debug(f"Updated backend {url} queue size to {queue_size}")
            else:
                # Backend is no longer available
                del self._available_backends[url]
                self._unavailable_backends.add(url)
                self._last_check_times[url] = current_time
                logging.warning(f"Backend {url} is no longer available")

    def _recheck_unavailable_backends(self, current_time):
        """Recheck unavailable backends to see if they've recovered."""
        unavailable_copy = list(self._unavailable_backends)
        for url in unavailable_copy:
            last_check = self._last_check_times.get(url, 0)
            if current_time - last_check >= self._health_check_interval:
                is_healthy, queue_size = self._check_backend_health(url)
                if is_healthy and queue_size is not None:
                    logging.info(f"Backend {url} is now available again with queue size {queue_size}")
                    self._unavailable_backends.remove(url)
                    self._available_backends[url] = queue_size
                    # Remove from last_check_times if it becomes available
                    if url in self._last_check_times:
                        del self._last_check_times[url]
                else:
                    # Update last check time even if still unavailable
                    self._last_check_times[url] = current_time


# Singleton instance of the backend manager
_backend_manager = None
_backend_manager_lock = threading.Lock()


def get_backend_manager():
    """Get the singleton backend manager instance."""
    global _backend_manager
    if _backend_manager is None:
        with _backend_manager_lock:
            if _backend_manager is None:
                _backend_manager = BackendManager()
    return _backend_manager


# -------------------------------------------------------------------
# Sandbox API Client Functions
# -------------------------------------------------------------------


def get_backend_url() -> str:
    """
    Get an available backend URL from the configured URLs.

    Returns:
        str: The backend URL to use

    Raises:
        BackendUnavailableError: If no available backends
    """
    manager = get_backend_manager()
    url = manager.get_available_backend()
    if url is None:
        raise BackendUnavailableError("No available backend servers. All backends are unreachable.")
    return url


def _make_api_request(
    method: str,
    endpoint: str,
    url: str,
    json_data=None,
    params=None,
    request_timeout: int = 60,
    client_error_codes: tuple[int, ...] = (400,),
) -> dict:
    """
    Helper function to make API requests with consistent error handling.

    Args:
        method: HTTP method ("get", "post")
        endpoint: API endpoint path
        url: Base backend URL
        json_data: JSON data for request body
        params: URL parameters
        request_timeout: Request timeout in seconds
        client_error_codes: HTTP status codes that should be treated as client errors

    Returns:
        JSON response data

    Raises:
        HTTPException: For client errors (400, 404, etc.)
        BackendUnavailableError: For backend unavailability
    """
    try:
        full_url = f"{url}/{endpoint}"
        request_fn = getattr(requests, method.lower())
        response = request_fn(full_url, json=json_data, params=params, timeout=request_timeout)

        # Let client errors pass through directly
        if response.status_code in client_error_codes:
            response.raise_for_status()
        elif not response.ok:
            # For server errors, mark backend as unavailable
            get_backend_manager().mark_backend_unavailable(url)
            raise BackendUnavailableError(f"Backend error: {response.status_code} {response.reason}")

        return response.json()
    except requests.ConnectionError as e:
        get_backend_manager().mark_backend_unavailable(url)
        raise BackendUnavailableError(f"Failed to connect to backend {url}: {e}") from e
    except requests.Timeout as e:
        get_backend_manager().mark_backend_unavailable(url)
        raise BackendUnavailableError(f"Backend timeout at {url}: {e}") from e
    except requests.RequestException as e:
        # Let client errors pass through from raise_for_status()
        if hasattr(e, "response") and e.response is not None and e.response.status_code in client_error_codes:
            raise
        # Other exceptions indicate backend issues
        get_backend_manager().mark_backend_unavailable(url)
        raise BackendUnavailableError(f"API request failed: {e}") from e


def submit_code(
    code: str,
    language: str,
    libraries: list[str],
    stdin: str | None = None,
    time_limit: int | None = None,
    memory_limit: int | None = None,
    request_timeout: int = 360,
    backend_base_url: str | None = None,
) -> str:
    """Submit code to the sandbox and return the submission ID."""
    url = backend_base_url if backend_base_url else get_backend_url()
    json_data = {
        "code": code,
        "language": language,
        "libraries": libraries,
        "stdin": stdin,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }
    response = _make_api_request("post", "submit", url, json_data, request_timeout=request_timeout)
    return response["submission_id"]


def submit_batch(
    requests_data: list[dict],
    request_timeout: int = 360,  # Increased timeout for batch
    backend_base_url: str | None = None,
) -> list[int]:
    """Submit a batch of code execution requests to the sandbox."""
    url = backend_base_url if backend_base_url else get_backend_url()
    json_data = {"requests": requests_data}
    response = _make_api_request("post", "submit_batch", url, json_data, request_timeout=request_timeout)
    return response["submission_ids"]


def get_submission_result(submission_id: str, request_timeout: int = 360, backend_base_url: str | None = None) -> dict:
    """Retrieve the sandbox result for a given submission ID."""
    url = backend_base_url if backend_base_url else get_backend_url()
    endpoint = f"submission/{submission_id}"
    return _make_api_request(
        "get",
        endpoint,
        url,
        request_timeout=request_timeout,
        client_error_codes=(400, 404),
    )


def get_batch_results(
    submission_ids: list[int],
    request_timeout: int = 120,  # Increased timeout for batch
    backend_base_url: str | None = None,
) -> list[dict]:
    """Retrieve the sandbox results for a batch of submission IDs."""
    url = backend_base_url if backend_base_url else get_backend_url()
    json_data = {"submission_ids": submission_ids}
    response = _make_api_request("post", "submissions", url, json_data, request_timeout=request_timeout)
    return response["results"]


def cancel_submission(submission_id: str, request_timeout: int = 60, backend_base_url: str | None = None) -> dict:
    """Cancel a submission that is currently waiting to be processed."""
    url = backend_base_url if backend_base_url else get_backend_url()
    endpoint = f"cancel/{submission_id}"
    return _make_api_request(
        "post",
        endpoint,
        url,
        request_timeout=request_timeout,
        client_error_codes=(400, 404),
    )

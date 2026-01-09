import concurrent.futures
from collections import Counter
from typing import Any

from .analysis import analyze_samples
from .run_tests import run_tests


def run_performance_benchmark(
    lang: str,
    solution: str,
    test_cases: list[dict[str, Any]],
    evaluator: str,
    test_runner: str | None = None,
    num_runs: int = 5,
    time_limit: int = 10,
    memory_limit: int = 1024,
    trim_ratio: float = 0.1,
    max_workers: int = 4,
) -> dict[str, Any]:
    """
    Runs a performance benchmark for a given solution against a set of test cases.

    This function executes the solution multiple times concurrently to gather runtime data,
    calculates the pass rate, and then computes a trimmed mean of the runtimes for the
    successful runs to provide a robust performance metric.

    Args:
        lang: The programming language of the solution.
        solution: The source code of the solution to be benchmarked.
        test_cases: A list of test cases, where each test case is a dictionary.
        evaluator: The function used to evaluate the correctness of the solution's output.
        test_runner: The function used to run the solution against a test case.
        num_runs: The number of times to run the solution against the test cases.
        time_limit: The time limit in seconds for each test run.
        memory_limit: The memory limit in megabytes for each test run.
        trimming_fraction: The fraction of runtimes to trim from each end before calculating the mean.
        max_workers: The maximum number of concurrent workers to use for running tests.

    Returns:
        A dictionary containing the pass rate and the trimmed mean runtime.
    """
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_tests,
                lang=lang,
                solution=solution,
                test_cases=test_cases,
                evaluator=evaluator,
                test_runner=test_runner,
                time_limit=time_limit,
                memory_limit=memory_limit,
                early_stop=False,  # We need to run all tests for performance
                raise_on_error=False,
                as_batch=False,
                polling_interval=10,
            )
            for _ in range(num_runs)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                # Handle exceptions from run_tests if necessary
                print(f"A test run failed with an exception: {e}")

    if not all_results:
        runtime_an = analyze_samples([], trim_ratio=trim_ratio)
        memory_an = analyze_samples([], trim_ratio=trim_ratio)
        integral_an = analyze_samples([], trim_ratio=trim_ratio)
        perf = {
            "original_n": 0,
            "n": 0,
            "runtime": runtime_an.get("trimmed_mean", float("inf")),
            "memory": memory_an.get("trimmed_mean", float("inf")),
            "integral": integral_an.get("trimmed_mean", float("inf")),
            "passed": False,
            "pass_rate": 0.0,
            "analysis": {
                "runtime": runtime_an,
                "memory": memory_an,
                "integral": integral_an,
            },
        }
        return {
            "performance_analysis": perf,
            "first_run_details": [],
            "failed_submission_exit_codes": [],
            "pass_rates": [],
            "pass_rate_consistent": True,
        }

    # Compute pass rate consistency across all runs
    pass_rates = []
    for test_case_results in all_results:
        total_cases_run = len(test_case_results)
        num_passed_run = sum(1 for tc in test_case_results if tc.get("passed", False))
        pr = num_passed_run / total_cases_run if total_cases_run > 0 else 0.0
        pass_rates.append(pr)

    pass_rate_consistent = len(set(pass_rates)) == 1

    # Calculate pass-rate by the majority vote
    pass_rate = Counter(pass_rates).most_common(1)[0][0]

    # Select the run whose pass rate matches the majority vote
    run_index = next((idx for idx, pr in enumerate(pass_rates) if pr == pass_rate), 0)
    first_run_results = all_results[run_index]

    # Collect detailed information about failed test cases from the first run
    failed_test_details = []
    for tc in first_run_results:
        if not tc.get("passed", False):
            failure_details = {
                "status": tc.get("status"),
                "text": tc.get("text"),
                "exit_code": tc.get("exit_code"),
                # Provide more context for debugging mismatches
                "input": tc.get("input"),
                "expected": tc.get("output"),
            }
            failed_test_details.append(failure_details)

    # Print brief pass-rate consistency summary
    try:
        pr_str = ", ".join(f"{pr:.2f}" for pr in pass_rates)
    except Exception:
        pr_str = ", ".join(str(pr) for pr in pass_rates)
    consistency_label = "consistent" if pass_rate_consistent else "inconsistent"
    if not pass_rate_consistent:
        print(
            f"Pass rate consistency across {len(all_results)} runs: {consistency_label} | pass_rates: [{pr_str}] | majority_based: {pass_rate:.2f}"
        )

    # Collect aggregated metrics per run only if all tests passed
    successful_runtimes: list[float] = []
    successful_memories: list[float] = []
    successful_integrals: list[float] = []
    if pass_rate == 1.0:
        for test_case_results in all_results:
            if bool(test_case_results) and all(tc.get("passed", False) for tc in test_case_results):
                total_runtime_ns = sum(tc.get("runtime", 0.0) for tc in test_case_results)
                total_runtime_s = total_runtime_ns / 1_000_000_000.0
                total_peak_memory_b = max(tc.get("memory", 0.0) for tc in test_case_results)
                total_peak_memory_mb = total_peak_memory_b / 1_000  # Convert to MB
                total_integral_mb_s = sum(tc.get("integral", 0.0) for tc in test_case_results)
                successful_runtimes.append(float(total_runtime_s))
                successful_memories.append(float(total_peak_memory_mb))
                successful_integrals.append(float(total_integral_mb_s))

    runtime_analysis = analyze_samples(successful_runtimes, trim_ratio=trim_ratio)
    memory_analysis = analyze_samples(successful_memories, trim_ratio=trim_ratio)
    integral_analysis = analyze_samples(successful_integrals, trim_ratio=trim_ratio)

    perf = {
        "original_n": num_runs,
        "n": len(successful_runtimes),
        "runtime": runtime_analysis.get("trimmed_mean", float("inf")),
        "memory": memory_analysis.get("trimmed_mean", float("inf")),
        "integral": integral_analysis.get("trimmed_mean", float("inf")),
        "passed": pass_rate == 1.0,
        "pass_rate": pass_rate,
        "analysis": {
            "runtime": runtime_analysis,
            "memory": memory_analysis,
            "integral": integral_analysis,
        },
    }

    return {
        "performance_analysis": perf,
        "first_run_details": first_run_results,
        "failed_test_details": failed_test_details,
        "pass_rates": pass_rates,
        "pass_rate_consistent": pass_rate_consistent,
    }

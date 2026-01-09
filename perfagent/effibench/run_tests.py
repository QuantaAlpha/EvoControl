import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backends.backend_utils import (
    BackendUnavailableError,
    cancel_submission,
    get_backend_url,
    get_batch_results,
    submit_batch,
    submit_code,
)
from .utils import (
    EFFIBENCH_REGISTRY,
    execute_with_timeout,
    get_full_code,
    get_sandbox_lang,
    materialize_function_from_code,
)


def raise_error(results: list[dict], code: str) -> None:
    for idx, result in enumerate(results):
        status = result["status"]
        test_input = result["input"] if len(result["input"]) < 2000 else result["input"][:2000] + "...truncated..."
        test_case_str = json.dumps({"input": test_input, "output": result["output"]})
        if status in ("waiting", "processing"):
            continue
        elif status == "timeout":
            raise TimeoutError(f"Test Case {idx + 1}: timed out.\nTest case: {test_case_str}")
        elif status == "oom":
            raise MemoryError(f"Test Case {idx + 1}: exceeded memory limit.\nTest case: {test_case_str}")
        elif status == "error" or result["exit_code"] != 0:
            program_output = result["text"]
            output_display = program_output if len(program_output) < 2000 else program_output[:2000] + "...truncated..."
            raise RuntimeError(
                f"Test Case {idx + 1}: runtime error.\nTest case: {test_case_str}\nProgram output: {output_display}\nCode: {code}\n"
            )


def postprocess_text(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    # More comprehensive regex to catch all ANSI escape sequences
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    text = text.replace("\r\r\n", "\n")
    text = text.replace("\r\n", "\n").strip()

    return text


class EvaluatorTimeoutError(TimeoutError):
    pass


def run_tests(
    lang: str,
    solution: str,
    test_cases: list,
    evaluator: str,
    test_runner: str | None = None,
    time_limit: int = 10,
    memory_limit: int = 1024,
    early_stop: bool = True,
    raise_on_error: bool = True,
    as_batch: bool = True,
    backend_retries: int = 5,
    eval_timeout: int = 10,
    polling_interval: int = 5,
    backend_retry_initial_wait: int = 10,
    backend_retry_max_wait: int = 60,
) -> list[dict]:
    code = get_full_code(lang, solution, test_runner) if test_runner else solution
    sandbox_lang = get_sandbox_lang(lang)
    libraries = EFFIBENCH_REGISTRY.get(lang, {}).get("packages", [])

    if lang == "java" and "class Codechef" in code:
        code = code.replace("Codechef", "Main")

    batch_requests = [
        {
            "code": code,
            "language": sandbox_lang,
            "libraries": libraries,
            "stdin": tc["input"],
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }
        for tc in test_cases
    ]
    evaluate = materialize_function_from_code(evaluator, "evaluate")
    for retry_count in range(backend_retries + 1):
        try:
            backend_base_url = get_backend_url()

            if as_batch:
                sids = submit_batch(batch_requests, backend_base_url=backend_base_url)
            else:
                sids = [None] * len(batch_requests)
                with ThreadPoolExecutor(len(batch_requests)) as executor:
                    future_to_tid = {
                        executor.submit(
                            submit_code,
                            code=req["code"],
                            language=req["language"],
                            libraries=req["libraries"],
                            stdin=req["stdin"],
                            time_limit=req["time_limit"],
                            memory_limit=req["memory_limit"],
                            backend_base_url=backend_base_url,
                        ): tid
                        for tid, req in enumerate(batch_requests)
                    }

                    for future in as_completed(future_to_tid):
                        sids[future_to_tid[future]] = future.result()

            sid_to_tid = {sid: tid for tid, sid in enumerate(sids)}
            all_results = [None] * len(test_cases)
            pending_ids = set(sids)

            while len(pending_ids):
                time.sleep(polling_interval)

                batch_results = get_batch_results(list(pending_ids), backend_base_url=backend_base_url)
                new_result_ids = set()
                for result_data in batch_results:
                    sid = result_data["submission_id"]
                    assert sid in pending_ids, f"Submission ID {sid} not in pending IDs"

                    if result_data["status"] not in ("waiting", "processing"):
                        all_results[sid_to_tid[sid]] = {
                            **result_data,
                            **test_cases[sid_to_tid[sid]],
                        }
                        pending_ids.remove(sid)
                        new_result_ids.add(sid)
                new_results = [all_results[sid_to_tid[sid]] for sid in new_result_ids]

                try:
                    raise_error(new_results, code)
                except Exception as e:
                    if raise_on_error or early_stop:
                        if pending_ids:
                            with ThreadPoolExecutor(max_workers=len(pending_ids)) as cancel_executor:
                                cancel_futures = [
                                    cancel_executor.submit(
                                        cancel_submission,
                                        sid_to_cancel,
                                        backend_base_url=backend_base_url,
                                    )
                                    for sid_to_cancel in pending_ids
                                ]
                                [None for f in as_completed(cancel_futures)]

                    if raise_on_error:
                        raise
                    if early_stop:
                        return [res for res in all_results if res]

                for idx, result_data in enumerate(new_results):
                    # Skip already evaluated results
                    if not result_data or "passed" in result_data:
                        continue

                    # For successful executions, evaluate the output
                    if result_data.get("status") == "done" and result_data.get("exit_code") == 0:
                        output = postprocess_text(result_data.get("text", ""))
                        expected = postprocess_text(result_data.get("output", ""))
                        try:
                            passed = execute_with_timeout(evaluate, eval_timeout, output, expected)
                            result_data["passed"] = passed

                            if not passed and raise_on_error:
                                test_input = result_data.get("input", "")
                                test_input_display = (
                                    test_input if len(test_input) < 2000 else test_input[:2000] + "...truncated..."
                                )
                                test_case_str = json.dumps(
                                    {
                                        "input": test_input_display,
                                        "output": result_data.get("output", ""),
                                    }
                                )
                                output_display = output if len(output) < 2000 else output[:2000] + "...truncated..."
                                raise AssertionError(
                                    f"Test Case {idx + 1}: failed.\nTest case: {test_case_str}\nProgram output: {output_display}\nCode: {code}\n"
                                )
                        except TimeoutError:
                            result_data["passed"] = False
                            if raise_on_error:
                                test_input = result_data.get("input", "")
                                test_input_display = (
                                    test_input if len(test_input) < 2000 else test_input[:2000] + "...truncated..."
                                )
                                test_case_str = json.dumps(
                                    {
                                        "input": test_input_display,
                                        "output": result_data.get("output", ""),
                                    }
                                )
                                raise EvaluatorTimeoutError(
                                    f"Solution Evaluator timed out after {eval_timeout} seconds.\nTest case: {test_case_str}\nCode: {code}\n"
                                )
                        except Exception:
                            result_data["passed"] = False

                    # Default to failed unless explicitly evaluated as passing
                    if "passed" not in result_data:
                        result_data["passed"] = False

            assert all(result_data for result_data in all_results), "Some test cases failed to run"
            return all_results

        except BackendUnavailableError as e:
            if retry_count < backend_retries:
                wait_s = backend_retry_initial_wait * (2**retry_count)
                if wait_s > backend_retry_max_wait:
                    wait_s = backend_retry_max_wait
                logging.warning(
                    f"Backend unavailable during execution (attempt {retry_count + 1}/{backend_retries + 1}): {e}. Retrying after {wait_s}s..."
                )
                time.sleep(wait_s)
                continue
            logging.error(f"All backends failed after {backend_retries} retries: {e}")
            raise
    return None

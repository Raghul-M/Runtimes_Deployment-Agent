"""QA Specialist for running ODH validation tests."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from . import SpecialistSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_qa_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Return the QA specialist agent and the supervisor-facing tool."""

    @tool
    def run_odh_tests() -> str:
        """
        Run the ODH model validation test suite using a staged kubeconfig under /tmp.

        This mirrors the manual pattern that worked:

            cp ~/.kube/config /tmp/kubeconfig-debug
            chmod 644 /tmp/kubeconfig-debug
            podman run --rm \
              -e KUBECONFIG=/root/.kube/config \
              -v /tmp/kubeconfig-debug:/root/.kube/config:Z \
              ...

        but adapted for the opendatahub-tests image, which runs as user 'odh'
        and expects to use /home/odh/.kube/config.

        While the tests run, this tool streams logs to stdout with a [QA] prefix.
        It returns a machine-readable status string:

            - 'QA_OK:<full logs>' on success
            - 'QA_ERROR:KUBECONFIG_MISSING ...'
            - 'QA_ERROR:KUBECONFIG_INVALID ...'
            - 'QA_ERROR:CLUSTER_UNREACHABLE ...'
            - 'QA_ERROR:TESTS_FAILED ...'
            - 'QA_ERROR:TIMEOUT ...'
            - 'QA_ERROR:RUNTIME_NOT_FOUND ...'
            - 'QA_ERROR:UNEXPECTED ...'
        """

        image = "quay.io/opendatahub/opendatahub-tests:latest"

        # 1. Resolve host kubeconfig (prefer $KUBECONFIG, then ~/.kube/config)
        host_kubeconfig = os.environ.get(
            "KUBECONFIG", os.path.expanduser("~/.kube/config")
        )
        host_kubeconfig_path = Path(host_kubeconfig)

        if not host_kubeconfig_path.exists():
            msg = f"QA_ERROR:KUBECONFIG_MISSING Host kubeconfig not found at {host_kubeconfig}"
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg

        # 2. Stage kubeconfig & results under /tmp (to avoid weird $HOME FS / xattrs)
        tmp_dir = Path(tempfile.mkdtemp(prefix="odh-tests-"))
        staged_kubeconfig = tmp_dir / "kubeconfig"
        results_dir = tmp_dir / "results"

        try:
            # Copy kubeconfig to a normal, writable filesystem
            shutil.copy2(host_kubeconfig_path, staged_kubeconfig)
            staged_kubeconfig.chmod(0o644)

            results_dir.mkdir(parents=True, exist_ok=True)
            results_dir.chmod(0o777)

            # 3. Build podman command
            cmd = [
                "podman",
                "run",
                "--rm",
                "-e",
                "KUBECONFIG=/home/odh/.kube/config",
                "-v",
                f"{staged_kubeconfig}:/home/odh/.kube/config:Z",
                "-v",
                f"{results_dir}:/home/odh/opendatahub-tests/results:Z",
                image,
            ]

            logger.info("Running ODH tests with command: %s", " ".join(map(str, cmd)))
            print("[QA] Starting ODH tests in container...", flush=True)

            # 4. Start process and stream logs
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout
                text=True,
                bufsize=1,
            )

            output_lines: list[str] = []
            start = time.time()
            timeout = 600  # seconds; adjust as needed

            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line)
                # Stream a snippet of logs to the user
                print(f"[QA] {line}", end="", flush=True)

                # Timeout check
                if time.time() - start > timeout:
                    proc.kill()
                    msg = (
                        f"QA_ERROR:TIMEOUT QA test suite did not complete within "
                        f"{timeout} seconds."
                    )
                    logger.error(msg)
                    print(f"\n[QA] {msg}\n", flush=True)
                    return msg

            proc.wait()
            full_output = "".join(output_lines)

            # 5. Classify result for the supervisor / decision layer
            if proc.returncode != 0:
                logger.error("ODH tests exited with code %s", proc.returncode)

                if "Invalid kube-config file" in full_output or "No configuration found" in full_output:
                    return "QA_ERROR:KUBECONFIG_INVALID " + full_output
                if "Trying to get client via new_client_from_config" in full_output:
                    return "QA_ERROR:CLUSTER_UNREACHABLE " + full_output

                return "QA_ERROR:TESTS_FAILED " + full_output

            print("[QA] ODH tests completed successfully.\n", flush=True)
            return "QA_OK:" + full_output

        except FileNotFoundError as e:
            logger.exception("podman not found when running ODH tests")
            msg = f"QA_ERROR:RUNTIME_NOT_FOUND podman not found or not executable: {e}"
            print(f"[QA] {msg}", flush=True)
            return msg
        except Exception as e:
            logger.exception("Unexpected error while running ODH tests")
            msg = f"QA_ERROR:UNEXPECTED {e}"
            print(f"[QA] {msg}", flush=True)
            return msg
        # Optional cleanup:
        # finally:
        #     shutil.rmtree(tmp_dir, ignore_errors=True)

    prompt = (
        "You are a QA Specialist responsible for validating machine learning model deployments "
        "and configurations on OpenShift / Kubernetes.\n\n"
        "You have access to a tool called `run_odh_tests` which runs the Opendatahub model "
        "validation test suite inside a container, and streams logs to the console.\n\n"
        "When a user asks to validate a deployment, or when you are invoked by the supervisor:\n"
        "1. Call `run_odh_tests`.\n"
        "2. Inspect its output string.\n"
        "3. Summarize whether QA passed or failed, and why.\n"
        "4. Provide clear, concise recommendations for next steps (e.g., fix kubeconfig, fix cluster access, "
        "   investigate failing tests, etc.).\n\n"
        "Never request kubeconfig contents or secrets from the user. Work only with the logs and status provided "
        "by the tool."
    )

    agent = create_agent(
        llm,
        tools=[run_odh_tests],
        system_prompt=prompt,
    )

    @tool
    def analyze_qa_results(request: str) -> str:
        """
        Analyze the QA test output and provide a summary.

        This is the supervisor-facing tool: the supervisor will send a natural language
        request (e.g., 'run QA and summarize the results'), and you should call the
        internal QA agent, which in turn decides when to use `run_odh_tests`.
        """
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return extract_text(result)

    analyze_qa_results.name = "analyze_qa_results"

    return SpecialistSpec(
        name="qa_specialist",
        agent=agent,
        tool=analyze_qa_results,
    )


__all__ = ["build_qa_specialist"]

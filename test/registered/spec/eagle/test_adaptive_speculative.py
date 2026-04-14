import json
import os
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=320, suite="stage-b-test-1-gpu-large")

HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)


def _resolve_local_model(*candidates: str, fallback: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return fallback


class TestAdaptiveSpeculativeServer(CustomTestCase):
    model = _resolve_local_model(
        "/models/shakechen/Llama-2-7b-chat-hf",
        "/root/models/shakechen/Llama-2-7b-chat-hf",
        fallback=DEFAULT_TARGET_MODEL_EAGLE,
    )
    draft_model = _resolve_local_model(
        "/models/lmsys/sglang-EAGLE-llama2-chat-7B",
        "/root/models/lmsys/sglang-EAGLE-llama2-chat-7B",
        fallback=DEFAULT_DRAFT_MODEL_EAGLE,
    )
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "candidate_steps": [1, 3],
                    "ema_alpha": 1.0,
                    "warmup_batches": 1,
                    "update_interval": 1,
                    "up_hysteresis": 0.0,
                },
                f,
            )
            cls.adaptive_config_path = f.name

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "triton",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                    "--skip-server-warmup",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _get_internal_state(self) -> dict:
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["internal_states"][0]

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": False,
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_adaptive_adjustment_switches_runtime_state_and_keeps_serving(self):
        flush_response = requests.get(self.base_url + "/flush_cache", timeout=30)
        self.assertEqual(flush_response.status_code, 200, flush_response.text)

        initial_state = self._get_internal_state()
        self.assertEqual(initial_state["speculative_num_steps"], 1)
        self.assertEqual(initial_state["speculative_num_draft_tokens"], 2)

        switched_state = initial_state
        switched = False
        for _ in range(4):
            result = self._generate(HIGH_ACCEPT_PROMPT)
            self.assertGreater(result["meta_info"].get("spec_verify_ct", 0), 0)

            switched_state = self._get_internal_state()
            print(
                "adaptive state:",
                switched_state["speculative_num_steps"],
                switched_state.get("avg_spec_accept_length"),
            )
            if switched_state["speculative_num_steps"] == 3:
                switched = True
                break

        self.assertTrue(
            switched, msg=f"Adaptive steps never switched: {switched_state}"
        )
        self.assertEqual(switched_state["speculative_num_draft_tokens"], 4)

        follow_up = self._generate("Today is a sunny day and I like", max_new_tokens=16)
        self.assertIsInstance(follow_up["text"], str)
        self.assertGreater(len(follow_up["text"]), 0)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from lead_agent import settings, storage
from lead_agent.clients import exa, http, llm
from lead_agent.errors import ApiError
from tests.fakes import TempOutputTestCase

ULTRA = {"effort": "ultra", "max_cost_dollars": 5, "max_duration_seconds": 1800}


class ExaRunTest(TempOutputTestCase):
    def setUp(self):
        super().setUp()
        self.patch(settings, "EXA_API_KEY", "test-key")
        self.patch(settings, "EXA_POLL_SECONDS", 0)
        self.patch(exa, "request_json", self.fake_request)
        self.requests = []
        self.statuses = ["running", "completed"]

    def fake_request(self, method, url, headers, body=None, timeout=0):
        self.requests.append((method, url, body))
        if method == "POST":
            return {"id": "agent_run_1", "status": "queued"}
        return {"id": "agent_run_1", "status": self.statuses.pop(0), "stopReason": "schema_satisfied", "output": {}}

    def test_starts_a_run_waits_and_cleans_up(self):
        run = exa.run_agent("demo", "query", {"type": "object"}, ULTRA, exclusion=[{"name": "Najad"}])

        self.assertEqual(run["status"], "completed")
        method, url, body = self.requests[0]
        self.assertEqual((method, url), ("POST", "https://api.exa.ai/agent/runs"))
        self.assertEqual(body["budget"], {"maxCostDollars": 5, "maxDurationSeconds": 1800})
        self.assertEqual(body["input"], {"exclusion": [{"name": "Najad"}]})
        self.assertFalse(storage.pending_exa_run_path("demo").exists())

    def test_resumes_a_saved_run_instead_of_paying_again(self):
        storage.write_json(storage.pending_exa_run_path("demo"), {"run_id": "agent_run_1"})
        exa.run_agent("demo", "query", {}, ULTRA)
        self.assertNotIn("POST", [method for method, _, _ in self.requests])

    def test_fixed_effort_runs_get_no_budget(self):
        exa.run_agent("demo", "query", {}, {"effort": "low", "max_cost_dollars": 5})
        self.assertNotIn("budget", self.requests[0][2])

    def test_a_failed_run_raises_and_is_forgotten(self):
        self.statuses = ["failed"]
        with self.assertRaises(ApiError):
            exa.run_agent("demo", "query", {}, ULTRA)
        self.assertFalse(storage.pending_exa_run_path("demo").exists())


class HttpRetryTest(unittest.TestCase):
    def test_busy_answers_are_retried(self):
        busy = mock.Mock(ok=False, status_code=429, text="slow down")
        fine = mock.Mock(ok=True, status_code=200)
        fine.json.return_value = {"ok": True}
        with mock.patch("requests.request", side_effect=[busy, fine]), mock.patch("time.sleep"):
            with self.assertLogs("lead_agent.clients.http", level="WARNING"):
                self.assertEqual(http.request_json("GET", "https://example.com", {}), {"ok": True})

    def test_client_errors_are_not_retried(self):
        bad = mock.Mock(ok=False, status_code=401, text="bad key")
        with mock.patch("requests.request", return_value=bad) as request:
            with self.assertRaises(ApiError) as caught:
                http.request_json("GET", "https://example.com", {})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(caught.exception.status_code, 401)


class ParseJsonTest(unittest.TestCase):
    def test_reads_json_inside_fences_or_text(self):
        self.assertEqual(llm.parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm.parse_json('Here you go: {"a": {"b": 2}} Done.'), {"a": {"b": 2}})
        with self.assertRaises(ValueError):
            llm.parse_json("no json here")


if __name__ == "__main__":
    unittest.main()

import unittest
from src.reports.validation_report import ReportingAgent

class TestReportingAgent(unittest.TestCase):
    def test_parse_html(self):
        agent = ReportingAgent("tests/sample_modelcar01.html", "dummy", "dummy")
        summary = agent.parse_html_report()  # Should match fake HTML structure for the test
        self.assertIsInstance(summary, dict)
        self.assertIn("passed", summary)

    def test_build_summary_message(self):
        summary = {"passed": 3, "failed": 2, "error": 0, "skipped": 1, "total": 6, "failed_cases": ["t1", "t2"], "error_cases": []}
        agent = ReportingAgent("dummy", "dummy", "dummy")
        msg = agent.build_summary_message(summary)
        self.assertIn("Failed Cases:", msg)

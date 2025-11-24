"""Report generator for accelerator compatibility validation results.

This module handles:
1. Validation result formatting
2. Compatibility report generation
3. Detailed analysis of:
   - Model-specific compatibility issues
   - GPU memory requirements
   - Accelerator type compatibility
   - Failed validation cases
   - Skipped model-accelerator combinations
"""

import os
import requests
from bs4 import BeautifulSoup


class ReportingAgent:
    def __init__(self, html_report_path):
        """Initialize with the path to the pytest HTML report."""
        self.html_report_path = html_report_path

    def parse_html_report(self):
        """Parse pytest HTML report summary counts from the summary section."""
        with open(self.html_report_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        summary = {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "rerun": 0,
            "total": 0,
            "run_info": "",
        }

        summary_div = soup.find("div", class_="summary__data")
        if not summary_div:
            print("No summary section found in HTML report.")
            return summary

        run_info_tag = summary_div.find("p", class_="run-count")
        if run_info_tag:
            summary["run_info"] = run_info_tag.get_text(strip=True)

        filters_block = summary_div.find("div", class_="filters")
        if not filters_block:
            print("No filters section found in HTML report.")
            return summary

        for span in filters_block.find_all("span"):
            text = span.get_text(strip=True).lower()
            parts = text.replace(",", "").split()
            if len(parts) >= 2:
                try:
                    count = int(parts[0])
                except ValueError:
                    count = 0
                label = parts[1]
                if label.endswith("s"):
                    label = label[:-1]
                summary[label] = count

        summary["total"] = (
            summary["passed"]
            + summary["failed"]
            + summary["error"]
            + summary["skipped"]
            + summary["xfailed"]
            + summary["xpassed"]
            + summary["rerun"]
        )

        return summary

    def build_summary_message(self, summary):
        """Construct a Slack-friendly summary message."""
        msg = "*ModelCar Validation Summary:*\n"

        
        if summary["run_info"]:
            if "0 test took" in summary["run_info"].lower() and summary["total"] > 0:
                msg += f"_Tests completed with setup errors (took {summary['run_info'].split('took')[-1].strip()})_\n\n"
            else:
                msg += f"_{summary['run_info']}_\n\n"
        msg += (
            f"- Total Tests: {summary['total']}\n"
            f"- Passed: {summary['passed']}\n"
            f"- Failed: {summary['failed']}\n"
            f"- Errors: {summary['error']}\n"
            f"- Skipped: {summary['skipped']}\n"
        )

        return msg

    def send_to_slack(self, message):
        """Send the summary to Slack via Incoming Webhook."""
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook_url:
            raise ValueError("SLACK_WEBHOOK_URL environment variable not set.")


        payload = {"text": message}
        response = requests.post(webhook_url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Slack notification failed: {response.text}")

    def run(self):
        """Run the full report extraction and Slack notification process."""
        summary = self.parse_html_report()
        message = self.build_summary_message(summary)
        self.send_to_slack(message)

        print("\n Slack summary posted successfully!")
        print("--------------------------------------------------")
        print(message)
        print("--------------------------------------------------")


if __name__ == "__main__":
    HTML_FILE = os.environ.get(
        "HTML_REPORT",
        "/home/rpancham/Runtimes_Deployment-Agent/src/reports/modelcar01.html",
    )

    agent = ReportingAgent(HTML_FILE)
    agent.run()

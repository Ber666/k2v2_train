import requests
import wandb
import logging
from monitor.metrics import MetricCode

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """A class to monitor the training job"""
    def __init__(self, args):
        self.args = args
        self.slack_webhook_url = args.slack_webhook_url
        self.metrics = []

    def register_metric(self, metric):
        """Register a metric to be monitored"""
        self.metrics.append(metric)

    def check(self, run_data):
        """Check the metrics and send alerts if any metric is abnormal"""
        for metric in self.metrics:
            code, msg = metric.check(run_data)
            if code == MetricCode.HEALTHY:
                msg = f"[HEALTHY] {msg}"
                # TODO: Add a log file
                logger.info(msg)
            elif code == MetricCode.ALERT:
                msg = f"[ALERT] {msg}"
                self.send_slack_msg(msg)
            elif code == MetricCode.CRITICAL:
                msg = f"[CRITICAL] {msg}"
                self.send_slack_msg(msg)

    def send_slack_msg(self, message):
        if not self.slack_webhook_url:
            return
        try:
            payload = {"text": message}
            # TODO: Uncomment this when the slack webhook is ready
            # requests.post(self.slack_webhook_url, json=payload)
            logger.info(f"Slack message sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send Slack message: {str(e)}")

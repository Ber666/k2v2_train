from abc import ABC, abstractmethod

class MetricCode:
    HEALTHY = 0
    ALERT = 1
    CRITICAL = 2

class Metric(ABC):
    def __init__(self, name, max_value=None, min_value=None, change_threshold=None):
        self.name = name
        self.max_value = max_value
        self.min_value = min_value
        self.change_threshold = change_threshold
        self.previous_value = None

    @abstractmethod
    def get_current_value(self, wandb_data):
        pass

    def check(self, wandb_data) -> tuple[MetricCode, str]:
        """Check if the metric is abnormal"""
        current_value = self.get_current_value(wandb_data)
        if current_value is None:
            # TODO: 'None' is not always healthy
            return MetricCode.HEALTHY, f"'{self.name}': No current data available. Current value: {current_value}"
        if self.previous_value is None:
            self.previous_value = current_value
            return MetricCode.HEALTHY, f"'{self.name}': Healthy. Current value: {current_value}"

        if self.is_value_abnormal(current_value):
            message = f"'{self.name}': Abnormal value detected. Current value: {current_value}"
            self.previous_value = current_value
            return MetricCode.ALERT, message

        if self.is_value_change_abnormal(self.previous_value, current_value):
            message = f"'{self.name}': Abnormal value change detected. Prev. value: {self.previous_value} -> Current value: {current_value}"
            self.previous_value = current_value
            return MetricCode.ALERT, message

        self.previous_value = current_value
        return MetricCode.HEALTHY, f"'{self.name}': Healthy. Current value: {current_value}"
    
    def is_value_abnormal(self, value):
        if self.max_value is not None and value > self.max_value:
            return True
        if self.min_value is not None and value < self.min_value:
            return True
        return False

    def is_value_change_abnormal(self, previous_value, current_value):
        # TODO: Can we reliably detect every abnormal change with current query interval?
        # Shall we use moving average?
        change_p = abs(current_value - previous_value) / previous_value
        if self.change_threshold is not None and change_p > self.change_threshold:
            return True
        return False


class JobStateMetric(Metric):
    def __init__(self, name):
        super().__init__(name, None)
        
    def get_current_value(self, wandb_data):
        return wandb_data.state

    def check(self, wandb_data):
        if wandb_data.state == "running":
            return MetricCode.HEALTHY, "'job status': The job is running."
        elif wandb_data.state == "finished":
            return MetricCode.CRITICAL, f"'job status': Training of job '{wandb_data.name}' was finished."
        else:
            return MetricCode.CRITICAL, f"'job status': Training of job '{wandb_data.name}' was terminated due to {wandb_data.state}."

class GPUPowerMetric(Metric):
    def get_current_value(self, wandb_data):
        return wandb_data.systemMetrics.get('system.gpu.0.powerWatts', None)

class GPUTemperatureMetric(Metric):
    def get_current_value(self, wandb_data):
        return wandb_data.systemMetrics.get('system.gpu.0.temp', None)

class GradNormMetric(Metric):
    def get_current_value(self, wandb_data):
        return wandb_data.summary.get('grad-norm', None)

class LMLossMetric(Metric):
    def get_current_value(self, wandb_data):
        return wandb_data.summary.get('lm loss', None)
import time
import argparse
import wandb
import logging

from monitor.training_monitor import TrainingMonitor
from monitor.metrics import (
    GPUPowerMetric, 
    GPUTemperatureMetric, 
    GradNormMetric, 
    LMLossMetric, 
    JobStateMetric
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_elapsed_time(seconds):
    days, seconds = divmod(int(seconds), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def get_args():
    parser = argparse.ArgumentParser()
    # WANDB
    parser.add_argument('--wandb-key', type=str, required=True)
    parser.add_argument('--project-name', type=str, required=True)
    # SLACK
    parser.add_argument('--slack_webhook_url', type=str, required=True)
    # INTERVAL
    parser.add_argument('--query-interval', type=int, default=600)
    # VALUES
    # JOB STATE
    parser.add_argument('--job-state-name', type=str, default="job state")
    # GPU POWER
    parser.add_argument('--watt-name', type=str, default="system.gpu.0.powerWatts")
    parser.add_argument('--watt-max', type=float, default=None)
    parser.add_argument('--watt-min', type=float, default=0.2)
    parser.add_argument('--watt-change-threshold', type=float, default=0.5)
    # GPU TEMPERATURE
    parser.add_argument('--temperature-name', type=str, default="system.gpu.0.temp")
    parser.add_argument('--temperature-max', type=float, default=70)
    parser.add_argument('--temperature-min', type=float, default=None)
    parser.add_argument('--temperature-change-threshold', type=float, default=50)
    # LM LOSS
    parser.add_argument('--lm-loss-name', type=str, default="lm loss")
    parser.add_argument('--lm-loss-max', type=float, default=None)
    parser.add_argument('--lm-loss-min', type=float, default=None)
    parser.add_argument('--lm-loss-change-threshold', type=float, default=2)
    # GRAD NORM
    parser.add_argument('--grad-norm-name', type=str, default="system.gpu.0.powerWatts")
    parser.add_argument('--grad-norm-max', type=float, default=10000)
    parser.add_argument('--grad-norm-min', type=float, default=None)
    parser.add_argument('--grad-norm-change-threshold', type=float, default=5000)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    wandb.login(key=args.wandb_key)
    monitor = TrainingMonitor(args)

    # Register metrics
    monitor.register_metric(JobStateMetric(name=args.job_state_name))
    monitor.register_metric(GPUPowerMetric(
        name=args.watt_name, 
        max_value=args.watt_max,
        min_value=args.watt_min,
        change_threshold=args.watt_change_threshold
        ))
    monitor.register_metric(GPUTemperatureMetric(
        name=args.temperature_name, 
        max_value=args.temperature_max,
        min_value=args.temperature_min,
        change_threshold=args.temperature_change_threshold
        ))
    monitor.register_metric(LMLossMetric(
        name=args.lm_loss_name, 
        max_value=args.lm_loss_max,
        min_value=args.lm_loss_min,
        change_threshold=args.lm_loss_change_threshold
        ))
    monitor.register_metric(GradNormMetric(
        name=args.grad_norm_name, 
        max_value=args.grad_norm_max,
        min_value=args.grad_norm_min,
        change_threshold=args.grad_norm_change_threshold
        ))
    
    start_time = time.time()
    check_counter = 0
    while True:
        formatted_time = format_elapsed_time(time.time() - start_time)
        logger.info(f"Check #{check_counter}: Elapsed time since start: {formatted_time}")
        api = wandb.Api()
        runs = api.runs(args.project_name, order="-created_at")
        running_runs = [r for r in runs if r.state == "running"]
        if len(running_runs) == 0:
            logger.info("No running jobs")
        else:
            for run in running_runs:
                monitor.check(run)
        time.sleep(args.query_interval)
        check_counter += 1


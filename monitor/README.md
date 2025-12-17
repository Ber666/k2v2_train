The `monitor` is a module to monitor the training job. The monitor will keep checking the metrics at a fixed interval (default 10 minutes) and send alerts to slack if any metric is abnormal. The log file is saved in the `monitor/log` directory.

## Run monitor
In the project root directory, run the following command to start the monitor:
```bash
bash examples/launch_monitor.sh $WANDB_API_KEY $WANDB_PROJECT_NAME $WANDB_RUN_NAME
```

# Add new monitor metrics
To add new monitor metrics, you need to implement a new metric class inheriting from the `Metric` class in `monitor/metrics.py`.
Two functions need to be implemented:
- `get_current_value(self, wandb_data)`: Get the current value of the metric.
- `check(self, wandb_data)`: Check if the metric is abnormal.

Finally, you need to register the new metric class in the `register_metrics` function in `monitor/training_monitor.py`.

## Candidate TODOs
- [ ] Deal with corner cases, e.g., 
    - [ ] when to start the monitor after the job is launched?
    - [ ] when to stop the monitor?
    - [ ] what if the job fails before the monitor starts
- [ ] Report the failure reason
- [ ] Add more necessarymetrics
- [ ] Add data source besides wandb, e.g., slurm job stats

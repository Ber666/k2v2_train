import wandb
import time
import argparse

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


previous_state = {
    "system.gpu.0.powerWatts": None,
    "system.gpu.0.temp": None,
    "grad-norm": None,
    "lm loss": None,
}

def is_value_abnormal(previous_value, current_value, threshold):
    if previous_value == 0: 
        return current_value != 0 
    
    change_p = abs(current_value - previous_value) / previous_value
    return change_p > threshold

def check_and_send(args, current_value, threshold, key):
    global previous_state
    if previous_state[key] == None:
        previous_state[key] = current_value
    elif is_value_abnormal(previous_state[key], current_value, threshold):
        send_to_slack(args, f"Abnormal state of {args.project_name}/{args.run_name}: {key} changes from {previous_state[key]} to {current_value}")
        previous_state[key] = current_value


def get_training_status(args):

    api = wandb.Api()
    run = api.run(f"{args.project_name}/{args.run_name}")

    # system_metrics = run.history(stream='systemMetrics')

    if run.state == "running":
        print("running")
        system_metrics = run.systemMetrics
        summary = run.summary


        # print(f"grad-norm: {summary['grad-norm']}")
        # print(f"lm loss: {summary['lm loss']}")
        # print(f"system.gpu.0.powerWatts: {system_metrics['system.gpu.0.powerWatts']}")
        # print(f"system.gpu.0.temp: {system_metrics['system.gpu.0.temp']}")

        # GPU watts
        check_and_send(args, system_metrics['system.gpu.0.powerWatts'], args.watt_threshold, "system.gpu.0.powerWatts")
        # GPU temperature
        check_and_send(args, system_metrics['system.gpu.0.temp'], args.temp_threshold, "system.gpu.0.temp")

        # loss spike
        check_and_send(args, summary['grad-norm'], args.grad_norm_threshold, "grad-norm")
        # grad norm spike
        check_and_send(args, summary['lm loss'], args.lm_loss_threshold, "lm loss")



    elif run.state == "finished":
        print("finished")
        send_to_slack(args, f"Training of {args.project_name}/{args.run_name} was finished.")
    else:
        print(run.state)
        send_to_slack(args, f"Training of {args.project_name}/{args.run_name} was terminated due to {run.state}.")


    return run.state

def send_to_slack(args, message: str):
    client = WebClient(token=args.client_token)
    channel_id = args.channel_id

    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        print(f"Message sent successfully: {response['message']['text']}")
    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-key', type=str, help="WANDB_API_KEY", required=True)
    parser.add_argument('--project-name', type=str, help="Project name in wandb", required=True, default="scaling-law")
    parser.add_argument('--run-name', type=str, help="run name in wandb", required=True, default="91ccpcsi")

    parser.add_argument('--client-token', type=str, help="Client token for chatbot in Slack", required=True)
    parser.add_argument('--channel-id', type=str, help="Indicates the channel to which the message will be sent", required=True)

    parser.add_argument('--query-interval', type=int, help="how often to query wandb", default=600)
    parser.add_argument('--grad-norm-threshold', type=float, help="When the difference between the result of this query and the previous one exceeds x%, an exception report is sent", default=0.5)
    parser.add_argument('--lm-loss-threshold', type=float, help="When the difference between the result of this query and the previous one exceeds x%, an exception report is sent", default=0.5)
    parser.add_argument('--watt-threshold', type=float, help="When the difference between the result of this query and the previous one exceeds x%, an exception report is sent", default=0.5)
    parser.add_argument('--temp-threshold', type=float, help="When the difference between the result of this query and the previous one exceeds x%, an exception report is sent", default=0.5)

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    wandb.login(key=args.wandb_key)

    while True:
        get_training_status(args)
        time.sleep(args.query_interval) 

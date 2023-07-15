from argparse import ArgumentParser
import json
from datetime import datetime


def get_flags():
    args = ArgumentParser(description='Tests on natural abstractions with CKA')
    args.add_argument(
        '--run_id', default=None, type=str,
        help='Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default')

    return args


def update_config(args: ArgumentParser):
    # Parse the command line arguments
    parsed_args = args.parse_args()
    
    # Load the existing config file
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Directly modify the config
    if parsed_args.run_id:
        config['run_id'] = parsed_args.run_id

    else:
        config['run_id'] = datetime.now().strftime('%Y%m%d%H%M%S')        

    return config

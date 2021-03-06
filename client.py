import os
import subprocess
import time
import torch
import zmq

import parameters
from consts import inference_times_name
from consts import label_final_name
from consts import out_client_name
from consts import out_final_name
from data.get_data import get_data
from utils.log_utils import create_logger
from utils.main_utils import round_array
from utils.time_utils import log_timing
from zmq_net.handle_numpy_array import recv_array
from zmq_net.handle_numpy_array import send_array


def run_client(args, data):
    port = args.port
    logger = create_logger(save_path='logs', file_type='client')
    logger.info(
        f"Client with port {port} execution: run private inference "
        f"(Step 1a of CaPC protocol).")
    if isinstance(port, list) or isinstance(port, tuple):
        logger.warn(
            "WARNING: list ports were passed. Only one should be passed.")
        port = port[0]  # only one port should be passed
    if args.batch_size > 1:
        raise ValueError('batch size > 1 not currently supported.')
    inference_start = time.time()
    print("Querying party: run inference (Step 1a)")

    # Connect to the server.
    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.hostname}:%s" % port)

    data = data.cpu().numpy()
    send_array(socket, data)
    r_rstar = recv_array(socket)

    inference_end = time.time()
    logger.info(
        f"Client (QP) with port {port} private inference (Step 1a) time: "
        f"{inference_end - inference_start}s")
    with open(inference_times_name, 'a') as outfile:
        outfile.write(str(inference_end - inference_start))
        outfile.write('\n')
    r_rstar = round_array(x=r_rstar, exp=args.round_exp)
    # logger.info(f"rounded r_rstar (r-r*): {array_str(r_rstar)}")
    with open(f'{out_client_name}{port}privacy.txt',
              'w') as outfile:  # r-r* vector saved (to be used in Step 1b)
        for val in r_rstar.flatten():
            outfile.write(f"{int(val)}\n")

    # do 2 party computation with the Answering Party
    msg = f"Client (QP) with port {port} starting secure 2PC for argmax " \
          f"(Step 1c) with its Answering Party (AP)."
    log_timing(stage='client:' + msg,
               log_file=args.log_timing_file)
    logger.info(msg)
    while not os.path.exists(
            f"{out_final_name}{port}.txt"):  # final_name = output
        process = subprocess.Popen(  # Step 1b of the protocol
            ['./mpc/bin/argmax', '2', '12345',
             # TODO: add ip address of the server
             f'{out_client_name}{port}privacy.txt'])
        process.wait()
    msg = f'Client (QP) with port {port} finished secure 2PC.'
    log_timing(stage=msg, log_file=args.log_timing_file)
    logger.info(msg)
    return r_rstar


def print_label():
    """Function to print final label after Step 3 is complete"""
    with open(f"{label_final_name}.txt", 'r') as file:
        label = file.read(1)
    logger = create_logger(save_path='logs', file_type='client')
    logger.info(f"Predicted label: {label}")


def main():
    args = parameters.get_args()

    train_loader, test_loader, train_dataset, test_dataset = get_data(args=args)

    data_id = args.indext # 0
    query = test_dataset[data_id][0]
    query = torch.unsqueeze(query, dim=0)
    correct_label = test_dataset[data_id][1]
    logger = create_logger(save_path='logs', file_type='client')
    logger.info(f'correct_label: {correct_label}')

    start_time = time.time()
    run_client(args=args, data=query)
    end_time = time.time()
    print(f'step 1a runtime: {end_time - start_time}s')
    log_timing('Client (QP) finished', log_file=args.log_timing_file)


if __name__ == "__main__":
    main()

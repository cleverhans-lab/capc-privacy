import numpy as np
import subprocess
import time
import zmq
import torch

import get_r_star
import parameters
from consts import (
    out_server_name,
    out_final_name,
    argmax_times_name,
    inference_no_network_times_name)
from models.get_models import get_model
from utils.log_utils import create_logger
from utils.main_utils import array_str
from utils.main_utils import round_array
from utils.time_utils import log_timing
from zmq_net.handle_numpy_array import recv_array, send_array


def get_rstar_server(max_logit, batch_size, num_classes, exp):
    """Return random vector r_star"""
    r_star = max_logit + np.random.uniform(
        low=-2 ** exp, high=2 ** exp, size=(batch_size, num_classes))
    return r_star


def run_server(args):
    logger = create_logger(save_path='logs', file_type='server')
    prefix_msg = f"Server (Answering Party AP) with port {args.port}: "
    logger.info(f"{prefix_msg}started Step 1a of the CaPC protocol).")

    model = get_model(dataset=args.dataset_name,
                      checkpoint_dir=args.checkpoint_dir,
                      device=args.device)
    logger.info(f"{prefix_msg}loaded model.")

    logger.info('Accept a query from a client.')
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % args.port)
    query = recv_array(socket)

    start_time = time.time()
    print(f"query shape before processing: {query.shape}")
    inference_start = time.time()
    print("Answering party: run private inference (Step 1a)")
    query = torch.from_numpy(query).to(args.device)
    y_hat = model(query)

    # print('r_star: ', FLAGS.r_star)
    logger.info(f"{prefix_msg}Step 1b: generate r* and send the share of "
                f"computed logits to QP.")
    r_star = get_r_star.get_rstar_server(
        # Generate a random vector needed in Step 1a.
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        seed=args.seed,
    ).flatten()
    print(f"rstar: {r_star}")

    y_hat = y_hat.detach().to('cpu').numpy()
    logger.info(f"y_hat: {y_hat}")
    # r - r* (subtract the random vector r* from logits) (to be used in Step 1b)
    r_rstar = y_hat - r_star

    send_array(socket, r_rstar)
    inference_end = time.time()
    logger.info(
        f"{prefix_msg}Inference time: {inference_end - inference_start}s")

    with open(inference_no_network_times_name, 'a') as outfile:
        outfile.write(str(inference_end - inference_start))
        outfile.write('\n')
    elapsed_time = time.time() - start_time
    print("total time(s)", np.round(elapsed_time, 3))

    msg = "Doing secure 2pc for argmax (Step 1c)."
    logger.info(f"{prefix_msg}{msg}")
    log_timing(stage='server:' + msg,
               log_file=args.log_timing_file)
    print('r_star (r*): ', array_str(r_star))
    r_star = round_array(x=r_star, exp=args.round_exp)
    print('rounded r_star (r*): ', array_str(r_star))
    if args.backend == 'HE_SEAL':
        argmax_time_start = time.time()
        with open(f'{out_server_name}{args.port}.txt',
                  'w') as outfile:  # party id
            # assume batch size of 1.
            for val in r_star.flatten():
                outfile.write(f"{int(val)}" + '\n')
        process = subprocess.Popen(
            ['./mpc/bin/argmax', '1', '12345',
             # TODO: add localhost for server
             # Calculate argmax of output logits (Step 1c)
             f'{out_server_name}{args.port}.txt',
             f'{out_final_name}{args.port}.txt'])  # noise, output  (s hat vectors, s vectors)
        process.wait()
        argmax_time_end = time.time()
        with open(argmax_times_name,
                  'a') as outfile:  # Save time taken for argmax computation to file.
            outfile.write(str(argmax_time_end - argmax_time_start))
            outfile.write("\n")
    msg = "finished 2PC for argmax (Step 1c)."
    log_timing(stage=f'server: {msg}',
               log_file=args.log_timing_file)
    logger.info(f"{prefix_msg}{msg}")


def main():
    args = parameters.get_args()
    run_server(args)


if __name__ == "__main__":
    main()

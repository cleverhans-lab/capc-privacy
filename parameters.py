import getpass

import argparse
import torch

from utils.time_utils import get_timestamp

DEFAULT_PORT = 8000


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("on", "yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("off", "no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def argument_parser():
    user = getpass.getuser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4567, help="random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='Number of workers to fetch the data.')
    parser.add_argument('--is_cuda', type=str2bool, default=True,
                        help='Is CUDA training enabled?')
    parser.add_argument('--final_call', type=int, default=0,
                        help='always 0 unless final call to client.')
    parser.add_argument('--encrypt_data_str', type=str, default="encrypt")
    parser.add_argument('--from_pytorch', type=int, default=1,
                        help='set to 1 to use pytorch bridge')
    parser.add_argument(
        "--backend", type=str, default="HE_SEAL", help="Name of backend to use")
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        # default="../../config/he_seal_ckks_config_N13_L4_gc_50.json",
        default='config/10.json',
        help="Filename containing json description of encryption parameters, "
             "or json description itself",
    )
    parser.add_argument(
        "--enable_client",
        type=str2bool,
        default=True,
        help="Enable the client")
    parser.add_argument(
        "--enable_gc",
        type=str2bool,
        default=True,
        help="Enable garbled circuits")
    parser.add_argument(
        "--mask_gc_inputs",
        type=str2bool,
        default=True,
        help="Mask garbled circuits inputs",
    )
    parser.add_argument(
        "--mask_gc_outputs",
        type=str2bool,
        default=True,
        help="Mask garbled circuits outputs",
    )
    parser.add_argument('--data_partition', type=str, default='test',
                        choices=['train', 'test'],
                        help='test or train partition.')
    parser.add_argument(
        "--num_gc_threads",
        type=int,
        default=20,
        help="Number of threads to run garbled circuits with",
    )
    parser.add_argument(
        "--input_node",
        type=str,
        default="import/input:0",  # input:0
        help="Tensor name of data input",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default="import/output/BiasAdd:0",  # local__model/dense_1/BiasAdd:0
        help="Tensor name of model output",
    )
    parser.add_argument(
        '--minibatch_id', type=int, default=0,
        help='which index in the minibatch to work on.'
    )
    parser.add_argument(
        '--indext', type=int, default=0,
        help='which index of the mnist test set to use as the query'
    )
    parser.add_argument(
        "--hostname", type=str, default="localhost", help="Hostname of server")
    parser.add_argument(
        "--pack_data",
        type=str2bool,
        default=True,
        help="Use plaintext packing on data")
    parser.add_argument(
        "--port", type=int,
        default=DEFAULT_PORT,
        help="Ports of server")
    parser.add_argument(
        "--rstar_exp",
        type=int,
        default=10,
        help='The exponent for 2 to generate the random r* from.',
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of possible classes in the classification task.",
    )
    parser.add_argument(
        '--checkpoint_dir', type=str,
        default=f'./models',
        # default=f'./architecture/',
        help='Path to the directory with all checkpoints.')

    parser.add_argument('--n_parties', type=int, default=1)  # , required=True)
    parser.add_argument(
        '--r_star',
        nargs='+',
        type=float,
        default=None,
        help="""For debug purposes: Each AP subtracts a vector of random 
        numbers r* from the logits r (this is done via the homomorphic 
        encryption). The encrypted result (r - r*) is sent back to the QP 
        (client). When QP decrypts the received result, it obtains (r - r*) in 
        plain text (note that this is not the plain result r). We can verify 
        that this was done correctly by computing (r - r*) + r* = r."""
    )
    parser.add_argument('--max_logit', default=36.0, type=float,
                        help='max logit found.')

    parser.add_argument("--query_ids", type=int, nargs='+')
    parser.add_argument(
        '--round_exp',
        type=int,
        default=None,
        help='Multiply r* and logits by 2^round_exp.'
    )
    parser.add_argument(
        '--dataset_path', type=str, default="", help='dataset to use.')
    parser.add_argument('--dataset_name', type=str, default='mnist',
                        help='name of dataset where queries came from.')
    parser.add_argument('--log_timing_file', type=str,
                        help='name of the global log timing file',
                        default=f'logs/log-timing-{get_timestamp()}.log')
    return parser


def get_args():
    args, unparsed = argument_parser().parse_known_args()

    torch.manual_seed(args.seed)

    args.is_cuda = args.is_cuda and torch.cuda.is_available()

    device = torch.device(
        "cuda" if args.is_cuda and torch.cuda.is_available() else "cpu")
    args.device = device

    return args

# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import ngraph_bridge
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from get_r_star import get_rstar_server
from utils.main_utils import array_str


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("on", "yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("off", "no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def server_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = (str(
        FLAGS.enable_client)).encode()
    if FLAGS.enable_client:
        server_config.parameter_map[tensor_param_name].s = b"client_input"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config


def main(FLAGS):
    num_classes = 10
    # rstar = tf.constant(np.arange(num_classes)[np.newaxis, :],
    #                     dtype=np.float32)
    rstar = get_rstar_server(
        batch_size=1,
        num_classes=num_classes,
        max_logit=None,
        exp=None,
    )
    print('rstar: ', array_str(rstar))
    b = tf.compat.v1.placeholder(
        tf.float32, shape=(1, num_classes), name="client_parameter_name")
    c = tf.compat.v1.placeholder(tf.float32, shape=(1, num_classes))
    f = c * (b - rstar)

    # Create config to load parameter b from client
    config = server_config_from_flags(FLAGS, b.name)
    print("config", config)

    with tf.compat.v1.Session(config=config) as sess:
        f_val = sess.run(f,
                         feed_dict={b: np.ones((1, num_classes)),
                                    c: np.ones((1, num_classes))})
        print("Result: ", f_val)

    print('rstar: ', array_str(rstar.squeeze()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--enable_client",
        type=str2bool,
        default=True,
        help="Enable the client")
    parser.add_argument(
        "--backend", type=str, default="HE_SEAL", help="Name of backend to use")
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        # default="../../config/he_seal_ckks_config_N11_L1.json",
        # default="he_seal_ckks_config_N11_L1_small.json",
        default="he_seal_ckks_config_N13_L5_gc_custom.json",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)

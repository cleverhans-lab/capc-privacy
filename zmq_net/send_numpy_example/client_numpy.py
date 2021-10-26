import zmq
import sys
import numpy as np
from zmq_net.handle_numpy_array import send_array, recv_array

port = "32000"
if len(sys.argv) > 1:
    port = sys.argv[1]
    int(port)

context = zmq.Context()
print("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:%s" % port)

#  Do 10 requests, waiting each time for a response
for request in range(1, 10):
    print("Sending request ", request, "...")
    send_array(socket, np.array([request, request]))
    #  Get the reply.
    message = recv_array(socket)
    print("Received reply ", request, "[", message, "]")

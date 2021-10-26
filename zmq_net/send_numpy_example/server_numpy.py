import zmq
import time
import sys
from zmq_net.handle_numpy_array import send_array, recv_array

port = "32000"
if len(sys.argv) > 1:
    port = sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    message = recv_array(socket)
    print("Received request: ", message)
    time.sleep(1)
    result = message + 1
    send_array(socket, result)

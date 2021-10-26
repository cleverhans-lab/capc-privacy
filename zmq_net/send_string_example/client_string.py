import zmq
import sys

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
    socket.send_string('Hello')
    #  Get the reply.
    message = socket.recv_string()
    print("Received reply ", request, "[", message, "]")

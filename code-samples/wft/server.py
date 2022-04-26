import asyncio
import websockets
import string
import random

recv_sockets = {}
send_sockets = {}


def code_generator():
    choices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    code = ""
    while code == "" or code in recv_sockets.keys():
        code = ""
        for x in range(0, 4):
            for y in range(0, 4):
                code += random.choice(choices)
            if x != 3:
                code += "-"
    return code


async def handler(websocket, path):
    connection_type = 0  # 0: unknown, 1: receiving, 2: sending (with valid code)
    connection_code = ""  # determined by generation or client depending on connection type
    init_response = await websocket.recv()  # "recv" for receiving end, "send [code]" for sending

    if init_response == "recv":  # receiving, needs to get code
        code = code_generator()
        recv_sockets[code] = websocket
        print("Added receiving connection, code " + code)

        connection_type = 1
        connection_code = code
        await websocket.send("confirm_recv " + code)  # inform client of code and successful connection

    if "send" in init_response:
        requested_code = init_response.replace("send ", "")
        if requested_code in recv_sockets.keys():  # receiving key exists
            send_sockets[requested_code] = websocket
            print("Added sending connection, code " + requested_code)

            connection_type = 2
            connection_code = requested_code
            await websocket.send("confirm_send")
        else:
            print("Invalid code received from sender, invalid code was " + requested_code)
            await websocket.send("invalid_code")  # client will be disconnected after this sends

    # At this point, the connection will be disconnected if it did not pass checks above

    if connection_type == 1:  # handling code for receiving end
        while not websocket.closed:
            async for message in websocket:
                if connection_code in send_sockets.keys():  # technically needs to be checked since client connects 2nd
                    try:
                        await send_sockets[connection_code].send(message)
                    except websockets.exceptions.ConnectionClosed:
                        del send_sockets[connection_code]
                        await websocket.send("client_disconnect")
                else:  # tried to send a message to a client that doesn't exist
                    await websocket.send("no_client")

        # socket closed
        del recv_sockets[connection_code]  # remove self from recv_sockets list
        print("Disconnect from receiver, code " + connection_code)

    if connection_type == 2:  # handling code for sending end
        while not websocket.closed:
            async for message in websocket:
                if connection_code in recv_sockets.keys():  # should be unless receiver disconnected
                    try:
                        await recv_sockets[connection_code].send(message)
                    except websockets.exceptions.ConnectionClosed:
                        del recv_sockets[connection_code]
                        await websocket.send("recv_disconnect")
                else:  # receiver was removed from the array somehow (probably disconnected)
                    await websocket.send("no_recv")

        # socket closed
        del send_sockets[connection_code]
        print("Disconnect from sender, code " + connection_code)


start_server = websockets.serve(handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

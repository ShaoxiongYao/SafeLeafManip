# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Robotiq, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Robotiq, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2012, Robotiq, Inc.
# Revision $Id$
#
# Modified from the original comModbusTcp by Kelsey Hawkins @ Georgia Tech


"""@package docstring
Module comModbusRtu: defines a class which communicates with Robotiq Grippers using the Modbus RTU protocol.

The module depends on pymodbus (https://code.google.com/p/pymodbus/) for the Modbus RTU client.
"""

from .robotiqmodbus.client.sync import ModbusSerialClient
from pymodbus.register_read_message import ReadHoldingRegistersResponse
from pymodbus.register_write_message import WriteMultipleRegistersResponse
from math import ceil
import time

def getCommand(cmd, value=0):
    """Get the serial command to send to the gripper."""

    # rACT = 1
    # rGTO = 1 
    # rATR = 0 
    # rPR = 0 
    # rSP = 255 
    # rFR = 150

    # #Build the command with each output variable
    # #To-Do: add verification that all variables are in their authorized range
    # message.append(rACT + (rGTO << 3) + (rATR << 4))
    # message.append(0)
    # message.append(0)
    # message.append(rPR)
    # message.append(rSP)
    # message.append(rFR)    

    #Initiate command as an empty list
    message = []

    if cmd == 'reset':
        # NOTE: reset is necessary at the first time
        return [0, 0, 0, 0, 0, 0] 
    elif cmd == 'activate':
        return [9, 0, 0, 0, 255, 150]
    elif cmd == 'close':
        return [9, 0, 0, 255, 255, 150]
    elif cmd == 'open':
        return [9, 0, 0, 0, 255, 150]
    elif cmd == 'position':
        return [9, 0, 0, value, 255, 150]

    return message 


class communication:
    def __init__(self, retry=False):
        """Setting the retry parameter to True will lead to retries until an answer is delivered."""
        self.client = None
        self.retry = retry

    def connectToDevice(self, device):
        """Connection to the client - the method takes the IP address (as a string, e.g. '192.168.1.11') as an argument."""
        self.client = ModbusSerialClient(
            method="rtu",
            port=device,
            stopbits=1,
            bytesize=8,
            baudrate=115200,
            timeout=0.05,
        )
        if not self.client.connect():
            print("Unable to connect to {}".format(device))
            return False
        return True

    def disconnectFromDevice(self):
        """Close connection"""
        self.client.close()

    def sendCommand(self, data):
        """Send a command to the Gripper - the method takes a list of uint8 as an argument. The meaning of each variable
        depends on the Gripper model (see support.robotiq.com for more details)"""
        # make sure data has an even number of elements
        if len(data) % 2 == 1:
            data.append(0)

        # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(0, len(data) // 2):
            message.append((data[2 * i] << 8) + data[2 * i + 1])

        if self.retry:
            while True:
                response = self.client.write_registers(0x03E8, message, unit=0x0009)
                if isinstance(response, WriteMultipleRegistersResponse):
                    break
        else:
            # To do!: Implement try/except
            self.client.write_registers(0x03E8, message, unit=0x0009)

    def getStatus(self, numBytes):
        """Sends a request to read, wait for the response and returns the Gripper status. The method gets the number of
        bytes to read as an argument"""
        numRegs = int(ceil(numBytes / 2.0))

        # Get status from the device
        if self.retry:
            while True:
                response = self.client.read_holding_registers(
                    0x07D0, numRegs, unit=0x0009
                )
                if isinstance(response, ReadHoldingRegistersResponse):
                    break
        else:
            # To do!: Implement try/except
            response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)
            if not response:
                print(
                    "Could not read registers on gripper. Please check connection and chosen device."
                )

        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        # Output the result
        return output

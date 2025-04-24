"""
Collection of transaction based abstractions
"""
import struct

from pymodbus.exceptions import ModbusIOException
from pymodbus.interfaces import IModbusFramer
from pymodbus.utilities import checkCRC, computeCRC


# --------------------------------------------------------------------------- #
# Modbus RTU Message
# --------------------------------------------------------------------------- #
class ModbusRtuFramer(IModbusFramer):
    """Modbus RTU Frame controller.
        [ Start Wait ] [Address ][ Function Code] [ Data ][ CRC ][  End Wait  ]
          3.5 chars     1b         1b               Nb      2b      3.5 chars
    Wait refers to the amount of time required to transmit at least x many
    characters.  In this case it is 3.5 characters.  Also, if we receive a
    wait of 1.5 characters at any point, we must trigger an error message.
    Also, it appears as though this message is little endian. The logic is
    simplified as the following::
        block-on-read:
            read until 3.5 delay
            check for errors
            decode
    The following table is a listing of the baud wait times for the specified
    baud rates::
        ------------------------------------------------------------------
         Baud  1.5c (18 bits)   3.5c (38 bits)
        ------------------------------------------------------------------
         1200   13333.3 us       31666.7 us
         4800    3333.3 us        7916.7 us
         9600    1666.7 us        3958.3 us
        19200     833.3 us        1979.2 us
        38400     416.7 us         989.6 us
        ------------------------------------------------------------------
        1 Byte = start + 8 bits + parity + stop = 11 bits
        (1/Baud)(bits) = delay seconds
    """

    def __init__(self, decoder):
        """Initialize a new instance of the framer.
        :param decoder: The decoder factory implementation to use
        """
        self.__buffer = b""
        self._buffer = b""
        self.__header = {}
        self.__hsize = 0x01
        self.__end = b"\x0d\x0a"
        self.__min_frame_size = 4
        self.decoder = decoder

    # ----------------------------------------------------------------------- #
    # Private Helper Functions
    # ----------------------------------------------------------------------- #
    def checkFrame(self):
        """Check if the next frame is available.
        Return True if we were successful.
        """
        try:
            self.populateHeader()
            frame_size = self.__header["len"]
            data = self.__buffer[: frame_size - 2]
            crc = self.__header["crc"]
            crc_val = (int(crc[0]) << 8) + int(crc[1])
            return checkCRC(data, crc_val)
        except (IndexError, KeyError, struct.error):
            return False

    def advanceFrame(self):
        """Skip over the current framed message.
        This allows us to skip over the current message after we have processed
        it or determined that it contains an error. It also has to reset the
        current frame header handle
        """
        self.__buffer = self.__buffer[self.__header["len"] :]
        self.__header = {}

    def resetFrame(self):  # pylint: disable=invalid-name
        """Reset the entire message frame.
        This allows us to skip over errors that may be in the stream.
        It is hard to know if we are simply out of sync or if there is
        an error in the stream as we have no way to check the start or
        end of the message (python just doesn't have the resolution to
        check for millisecond delays).
        """
        self.__buffer = b""
        self.__header = {}

    def isFrameReady(self):
        """Check if we should continue decode logic.
        This is meant to be used in a while loop in the decoding phase to let
        the decoder know that there is still data in the buffer.
        :returns: True if ready, False otherwise
        """
        return len(self.__buffer) > self.__hsize

    def populateHeader(self):  # pylint: disable=invalid-name
        """Try to set the headers `uid`, `len` and `crc`.
        This method examines `self.__buffer` and writes meta
        information into `self.__header`. It calculates only the
        values for headers that are not already in the dictionary.
        Beware that this method will raise an IndexError if
        `self.__buffer` is not yet long enough.
        """
        data = self.__buffer
        self.__header["uid"] = int(data[0])
        func_code = int(data[1])
        pdu_class = self.decoder.lookupPduClass(func_code)
        size = pdu_class.calculateRtuFrameSize(data)
        self.__header["len"] = size

        if len(data) < size:
            # crc yet not available
            raise IndexError
        self.__header["crc"] = data[size - 2 : size]

    def addToFrame(self, message):
        """Add the received data to the buffer handle.
        :param message: The most recent packet
        """
        self.__buffer += message

    def getFrame(self):
        """Get the next frame from the buffer.
        :returns: The frame data or ''
        """
        start = self.__hsize
        end = self.__header["len"] - 2
        buffer = self.__buffer[start:end]
        if end > 0:
            return buffer
        return b""

    def populateResult(self, result):
        """Populate the modbus result header.
        The serial packets do not have any header information
        that is copied.
        :param result: The response packet
        """
        result.unit_id = self.__header["uid"]

    # ----------------------------------------------------------------------- #
    # Public Member Functions
    # ----------------------------------------------------------------------- #
    def processIncomingPacket(self, data, callback):  # pylint: disable=arguments-differ
        """Process new packet pattern.
        This takes in a new request packet, adds it to the current
        packet stream, and performs framing on it. That is, checks
        for complete messages, and once found, will process all that
        exist.  This handles the case when we read N + 1 or 1 // N
        messages at a time instead of 1.
        The processed and decoded messages are pushed to the callback
        function to process and send.
        :param data: The new packet data
        :param callback: The function to send results to
        """
        self.addToFrame(data)
        while self.isFrameReady():
            if self.checkFrame():
                result = self.decoder.decode(self.getFrame())
                if result is None:
                    raise ModbusIOException("Unable to decode response")
                self.populateResult(result)
                self.advanceFrame()
                callback(result)  # defer or push to a thread?
            else:
                self.resetFrame()  # clear possible errors

    def buildPacket(self, message):
        """Create a ready to send modbus packet.
        :param message: The populated request/response to send
        """
        data = message.encode()
        unit_id = 9
        function_code = 16
        packet = struct.pack(">BB", unit_id, function_code) + data
        packet += struct.pack(">H", computeCRC(packet))
        return packet

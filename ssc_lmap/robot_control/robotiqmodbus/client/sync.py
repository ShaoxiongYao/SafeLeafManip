"""Sync client."""
import logging
import serial
from pymodbus.constants import Defaults
from pymodbus.factory import ClientDecoder
from pymodbus.exceptions import NotImplementedException, ParameterException
from pymodbus.exceptions import ConnectionException
from .transaction import FifoTransactionManager
from pymodbus.transaction import DictTransactionManager
from pymodbus.transaction import ModbusSocketFramer, ModbusBinaryFramer
from pymodbus.transaction import ModbusAsciiFramer
from .robotiqrtuframer import ModbusRtuFramer
from pymodbus.client.mixin import ModbusClientMixin

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The Synchronous Clients
# --------------------------------------------------------------------------- #


class BaseModbusClient(ModbusClientMixin):
    """Interface for a modbus synchronous client.

    Defined here are all the methods for performing the related request
    methods.  Derived classes simply need to implement the transport
    methods and set the correct framer.
    """

    def __init__(self, framer):
        """Initialize a client instance

        :param framer: The modbus framer implementation to use
        """
        self.framer = framer
        if isinstance(framer, ModbusSocketFramer):
            self.transaction = DictTransactionManager(self)
        else:
            self.transaction = FifoTransactionManager(self)

    # ----------------------------------------------------------------------- #
    # Client interface
    # ----------------------------------------------------------------------- #
    def connect(self):
        """Connect to the modbus remote host.

        :returns: True if connection succeeded, False otherwise
        """
        raise NotImplementedException("Method not implemented by derived class")

    def close(self):
        """Close the underlying socket connection."""
        pass

    def _send(self, request):  # pylint: disable=no-self-use
        """Send data on the underlying socket.

        :param request: The encoded request to send
        :return: The number of bytes written
        """
        raise NotImplementedException("Method not implemented by derived class")

    def _recv(self, size):  # pylint: disable=no-self-use
        """Read data from the underlying descriptor.

        :param size: The number of bytes to read
        :return: The bytes read
        """
        raise NotImplementedException("Method not implemented by derived class")

    # ----------------------------------------------------------------------- #
    # Modbus client methods
    # ----------------------------------------------------------------------- #
    def execute(self, request=None):
        """Execute.

        :param request: The request to process
        :returns: The result of the request execution
        """
        if not self.connect():
            raise ConnectionException("Failed to connect[%s]" % (self.__str__()))
        return self.transaction.execute(request)

    # ----------------------------------------------------------------------- #
    # The magic methods
    # ----------------------------------------------------------------------- #
    def __enter__(self):
        """Implement the client with enter block.

        :returns: The current instance of the client
        """
        if not self.connect():
            raise ConnectionException("Failed to connect[%s]" % (self.__str__()))
        return self

    def __exit__(self, klass, value, traceback):
        """Implement the client with exit block."""
        self.close()

    def __str__(self):
        """Build a string representation of the connection.

        :returns: The string representation
        """
        return "Null Transport"


# --------------------------------------------------------------------------- #
# Modbus Serial Client Transport Implementation
# --------------------------------------------------------------------------- #


class ModbusSerialClient(
    BaseModbusClient
):  # pylint: disable=too-many-instance-attributes
    """Implementation of a modbus serial client."""

    def __init__(self, method="ascii", **kwargs):
        """Initialize a serial client instance.

        The methods to connect are::

          - ascii
          - rtu
          - binary

        :param method: The method to use for connection
        :param port: The serial port to attach to
        :param stopbits: The number of stop bits to use
        :param bytesize: The bytesize of the serial messages
        :param parity: Which kind of parity to use
        :param baudrate: The baud rate to use for the serial device
        :param timeout: The timeout between serial requests (default 3s)
        """
        self.method = method
        self.socket = None
        BaseModbusClient.__init__(self, self.__implementation(method))

        self.port = kwargs.get("port", 0)
        self.stopbits = kwargs.get("stopbits", Defaults.Stopbits)
        self.bytesize = kwargs.get("bytesize", Defaults.Bytesize)
        self.parity = kwargs.get("parity", Defaults.Parity)
        self.baudrate = kwargs.get("baudrate", Defaults.Baudrate)
        self.timeout = kwargs.get("timeout", Defaults.Timeout)

    @staticmethod
    def __implementation(method):
        """Return the requested framer.

        :method: The serial framer to instantiate
        :returns: The requested serial framer
        """
        method = method.lower()
        if method == "ascii":
            return ModbusAsciiFramer(ClientDecoder())
        if method == "rtu":
            return ModbusRtuFramer(ClientDecoder())
        if method == "binary":
            return ModbusBinaryFramer(ClientDecoder())
        if method == "socket":
            return ModbusSocketFramer(ClientDecoder())
        raise ParameterException("Invalid framer method requested")

    def connect(self):
        """Connect to the modbus serial server.

        :returns: True if connection succeeded, False otherwise
        """
        if self.socket:
            return True
        try:
            self.socket = serial.Serial(
                port=self.port,
                timeout=self.timeout,
                bytesize=self.bytesize,
                stopbits=self.stopbits,
                baudrate=self.baudrate,
                parity=self.parity,
            )
        except serial.SerialException as msg:
            _logger.error(msg)
            self.close()
        return self.socket is not None

    def close(self):
        """Close the underlying socket connection."""
        if self.socket:
            self.socket.close()
        self.socket = None

    def _send(self, request):
        """Send data on the underlying socket.

        :param request: The encoded request to send
        :return: The number of bytes written
        """
        if not self.socket:
            raise ConnectionException(self.__str__())
        if request:
            return self.socket.write(request)
        return 0

    def _recv(self, size):
        """Read data from the underlying descriptor.

        :param size: The number of bytes to read
        :return: The bytes read
        """
        if not self.socket:
            raise ConnectionException(self.__str__())
        return self.socket.read(size)

    def __str__(self):
        """Build a string representation of the connection.

        :returns: The string representation
        """
        return "%s baud[%s]" % (self.method, self.baudrate)


# --------------------------------------------------------------------------- #
# Exported symbols
# --------------------------------------------------------------------------- #
__all__ = ["ModbusSerialClient"]

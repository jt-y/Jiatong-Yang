from VISA_Driver import VISA_Driver
from InstrumentConfig import InstrumentQuantity
import numpy as np

class Error(Exception):
    pass

class Driver(VISA_Driver):
    """ This class implements the DSP7265 lock-in amplifier"""

    def performOpen(self, options={}):
            """Perform the operation of opening the instrument connection"""

            # calling the generic VISA open to make sure we have a connection
            VISA_Driver.performOpen(self, options=options)

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """Perform the Set Value instrument operation. This function should
        return the actual value set by the instrument"""

        if quant.name in ('Oscillator amplitude', 'Oscillator frequency'):
            value = float(value)
            self.writeAndLog('{:s} {:e}'.format(quant.set_cmd, value))
            return value
        else:
            pass
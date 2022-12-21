import pyvisa
import time
from pyvisa.constants import StopBits, Parity


class lockinAmp():

    def __init__(self, GPIBAddress = 12):

        # define resource manager
        self.rm = pyvisa.ResourceManager()
        #self.rm = pyvisa.ResourceManager('C:/Windows/sysWOW64/visa32.dll') # 64 bit windows

        # define instrument and set terminators
        self.sr = self.rm.open_resource('GPIB0::' + str(GPIBAddress) + '::INSTR') #SR7265 GPIB address 12
        self.sr.read_termination = '\r\n'
        self.sr.write_termination = '\r\n'

        # print identification to test connection
        print(self.sr.query("*IDN?"))

        # initialize parameters using the floating point mode
        self.amplitude = 0.1 # [V]
        self.frequency = 2000 # [Hz]

        self.sr.write("OA %d" %(self.amplitude*1.0e6))
        self.sr.write("OF. %d" %self.frequency)

        self.sr.write("IMODE 0") # Current mode off, voltage mode input enabled
        self.sr.write("VMODE 1") # A input only
        self.sr.write("SEN 27") # sensitivity = 1V
        self.sr.write("TC 11") # time constant = 100 ms


    def __str__(self):
        return "initialized"


    def outputSignal(self, amp, freq):
        '''Set output frequency (Hz) and amplitude (V)'''
        
        self.amplitude = amp*1e6 #Oscillator output Vrms from 0 to 5V
        self.frequency = freq #Oscillator output frequency from 0 to 250kHz
        self.sr.write("OA %d" %self.amplitude)
        self.sr.write("OF. %d" %self.frequency)


    def sensitivity(self, mode):
        '''Set sensitivity of the lock-in'''

        self.sr.write("SEN %d" %mode)
        print('Sensitivity: '+self.sr.query('SEN.')+' V')

    def timeConst(self, mode):
        '''Set time constant of the lock-in'''

        self.sr.write("TC %d" %mode)
        print('Time constant: '+self.sr.query("TC.")+' s')
    
    def dataReturn(self):
        '''Return frequency [Hz], amplitude [V], and phase [deg]'''
        freq = float(self.sr.query("FRQ."))
        amp = float((self.sr.query("MAG.")).rstrip('\x00'))
        phase = float((self.sr.query("PHA.")).rstrip('\x00'))

        return [freq, amp, phase]
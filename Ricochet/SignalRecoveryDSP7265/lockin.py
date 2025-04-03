import pyvisa
import time
import numpy as np
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


    def sensitivity(self, sens):
        '''Set sensitivity of the lock-in'''

        sensitivity = np.array([2e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1])
        mask = sensitivity >= sens
        mode = np.argmax(mask)+1
        self.sr.write("SEN %d" %mode)
        print('Sensitivity set to: '+self.sr.query('SEN.')+' V')

    def timeConst(self, mode):
        '''Set time constant of the lock-in'''
        
        time_const = np.array([10e-6, 20e-6, 40e-6, 80e-6, 160e-6, 320e-6, 640e-6, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3])
        mask = time_const >= mode
        mode = np.argmax(mask)
        self.sr.write("TC %d" %mode)
        print('Time constant set to: '+self.sr.query("TC.")+' s')
    
    def dataReturn(self):
        '''Return frequency [Hz], amplitude [V], and phase [deg]'''
        freq = float(self.sr.query("FRQ."))
        amp = float((self.sr.query("MAG.")).rstrip('\x00'))
        phase = float((self.sr.query("PHA.")).rstrip('\x00'))

        return freq, amp, phase
    
    def freq_scan(self, amp_out, freqs):
        '''Scan a certain frequency range and return the amplitude and phase'''

        amplitude = np.zeros_like(freqs)
        phase = np.zeros_like(freqs)

        for i in range(len(freqs)):
            self.outputSignal(amp_out, freqs[i])
            time.sleep(self.sr.query("TC.")*10)
            _, amplitude[i], phase[i] = self.dataReturn()
        
        return amplitude, phase
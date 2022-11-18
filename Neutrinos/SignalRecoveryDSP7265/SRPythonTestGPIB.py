# opens a connection to the lock-in via USB of sModelNumber (string: '7230', '7270',
# or '7124') and sSerialNumber (string)
def Connection_Open_USB(sModelNumber, sSerialNumber):
    print('Open connection via USB')
    if (sModelNumber == '7270'):
       inst = rm.open_resource('USB0::0x0A2D::0x001B::' + sSerialNumber + '::RAW')
    if (sModelNumber == '7230'):
       inst = rm.open_resource('USB0::0x0A2D::0x0027::' + sSerialNumber + '::RAW')
    if (sModelNumber == '7124'):
       inst = rm.open_resource('USB0::0x0A2D::0x0018::' + sSerialNumber + '::RAW')
    return inst

# sends a write command string sCmd via USB to the lock-in already opened as inst
# and returns the resulting response string, status nad overload bytes
def Inst_Query_Command_USB(inst, sCmd):
    print('Send query command: ' + sCmd)
    inst.write_raw(sCmd)
    sResponse = inst.read()
    # read the status and overload bytes
    nStb = bytes((sResponse[len(sResponse)-2:len(sResponse)-1:]),'utf-8')
    nOvb = bytes((sResponse[len(sResponse)-1:len(sResponse):]),'utf-8')
    nStatusByte = int(nStb[0])
    # mask out bits 4, 5 & 6 which are not consistent across all instruments
    nStatusByte = nStatusByte & 143
    nOverloadByte = int(nOvb[0])
    # strip the returned response of the line feed, status & overload bytes, and 
    # the null terminator
    sResponse = sResponse[0:len(sResponse)-4:]
    # return the response from the instrument, the status byte, and the overload byte
    return sResponse, nStatusByte, nOverloadByte

# opens a connection to the lock-in via Ethernet to the specified IP address.
# Socket 50001 is used
def Connection_Open_Ethernet(sIPAddress):
    print('Open connection via Ethernet')
    inst = rm.open_resource('TCPIP0::' + sIPAddress + '::50001::SOCKET')
    return inst

# sends a write command string sCmd via Ethernet to the lock-in already opened 
# as inst and returns the resulting response string and status byte
def Inst_Query_Command_Ethernet(inst, sCmd):
    print('Send query command: ' + sCmd )
    inst.write_raw(sCmd +'\r')
    sResponse = ''
    # read until a CR is received
    sEcho = inst.read_bytes(1).decode('utf8')
    while (sEcho != '\r'):
        sResponse = sResponse + sEcho
        sEcho = inst.read_bytes(1).decode('utf8')
    # now read the status byte
    inst.write_raw('ST\r')
    sStatusByte = ''
    # read until a CR is received
    sEcho = inst.read_bytes(1).decode('utf8')
    while (sEcho != '\r'):
        sStatusByte = sStatusByte + sEcho
        sEcho = inst.read_bytes(1).decode('utf8')
    nStatusByte = int(sStatusByte)
    # mask out bits 4, 5 & 6 which are not consistent across all instruments
    nStatusByte = nStatusByte & 143
    sResponse = sResponse.replace('\n','')
    sResponse = sResponse.replace('\r','')
    # return the response from the instrument and the status byte
    return sResponse, nStatusByte

# opens a connection to the lock-in via GPIB of sGBIBAddress (string)
def Connection_Open_GPIB(sGBIBAddress):
    print('Open connection via GPIB')
    inst = rm.open_resource('GPIB0::' + sGBIBAddress + '::INSTR')
    return inst

# sends a write command string sCmd via GPIB to the lock-in already opened as inst
# and returns the resulting response string and status byte
def Inst_Query_Command_GPIB(inst, sCmd):
    print('Send query command: ' + sCmd)
    inst.write(sCmd)
    sResponse =  ''
    # repeatedly read status byte until bit 7 is asserted = data available or 
    # until bit 1 = command done is asserted
    nStatusByte = int(inst.stb)    
    while (nStatusByte & 0x80 != 0x80):
        nStatusByte = int(inst.stb)
        if (nStatusByte & 0x01 == 0x01):
            break
    if (nStatusByte & 0x80 == 0x80):
        # data available, read it back
        sResponse = inst.read()
        # strip the CR LF terminator
        sResponse = sResponse[0:len(sResponse)-2:]
    # read status byte until bit 1 is asserted = command done
    while (nStatusByte & 0x01 != 0x01):
        nStatusByte = int(inst.stb)
    # mask out bits 4, 5 & 6 which are not consistent across all instruments
    nStatusByte = nStatusByte & 143
    # return the response from the instrument and the status byte
    return sResponse, nStatusByte

# opens a connection to the lock-in via RS232 of sPort, sBaudRate (strings)
def Connection_Open_RS232(sPort, sBaudRate):
    print('Open connection via RS232')
    inst = rm.open_resource(sPort)
    inst.baud_rate = int(sBaudRate)
    inst.parity = Parity.even
    inst.data_bits = 7
    return inst
   
# sends a write command string sCmd via RS232 to the lock-in already opened as inst
# and returns the resulting response string and status byte
def Inst_Query_Command_RS232(inst, sCmd):
    print('Send query command: ' + sCmd)
    # serial port commands need sending one character at a time and checking for 
    # handshake
    for i in range(len(sCmd)):
        inst.write_raw(sCmd[i])
        sEcho = inst.read_bytes(1).decode('utf8')
    # write the terminator
    inst.write_raw('\r')
    sResponse = ''
    # read until recieve a prompt of ? or *
    sEcho = inst.read_bytes(1).decode('utf8')
    while (sEcho != '?') and (sEcho != '*'):
        sResponse = sResponse + sEcho
        sEcho = inst.read_bytes(1).decode('utf8')
    sResponse = sResponse.replace('\n','')
    sResponse = sResponse.replace('\r','')
    # set returned status byte to Comamnd Done
    nStatusByte = 1
    if (sEcho == '*'):
       nStatusByte = 1
    if ((sEcho == '?') & (sCmd != 'ST')):
        # send the status command to get instrument status except if this is being
        # called by a ST command itself
        sStatus = Inst_Query_Command_RS232(inst, 'ST')
        nStatusByte = int(sStatus[0])
    # mask out bits 4, 5 & 6 which are not consistent across all instruments
    nStatusByte = nStatusByte & 143
    # return the response from the instrument and the status byte
    return sResponse, nStatusByte  

# closes the open resource (use for USB, GPIB, RS232, and Ethernet)
def Connection_Close(inst):
    print('Close connection')
    inst.before_close()
    return_status = inst.close()
    return return_status


def Print_Status_Byte(nStatusByte):
    if (nStatusByte & 1 == 1):
       print('Command Done')
    if (nStatusByte & 2 == 2):
       print('Invalid command')
    if (nStatusByte & 4 == 4):
       print('Command parameter error')
    if (nStatusByte & 8 == 8):
       print('Reference unlock')
    # bits 4, 5 and 6 are instrument model number dependent so are
    # not decoded here
    if (nStatusByte & 128 == 128):
       print('Data Available')

def Print_72XXOverload_Byte(nOverloadByte):
    if (nOverloadByte & 1 == 1):
       print('X(1) output overload')
    if (nOverloadByte & 2 == 2):
       print('Y(1) output overload')
    if (nOverloadByte & 4 == 4):
       print('X2 output overload')
    if (nOverloadByte & 8 == 8):
       print('Y2 output overload')
    if (nOverloadByte & 16 == 16):
       print('CH1 output overload')
    if (nOverloadByte & 32 == 32):
       print('CH2 output overload')
    if (nOverloadByte & 64 == 64):
       print('CH3 output overload')
    if (nOverloadByte & 128 == 128):
       print('CH4 output overload')


# main code starts here     
import pyvisa
from pyvisa.constants import StopBits, Parity
rm = pyvisa.ResourceManager('C:/Windows/System32/visa32.dll') # 32 bit windows
# rm = pyvisa.ResourceManager('C:/Windows/sysWOW64/visa32.dll') # 64 bit windows


# Print the list of VISA resources on this computer
rm.list_resources()
print('The VISA resourses present on this computer are: ')
print(rm.list_resources('?*'))





# Demonstration of GPIB communications
print('Demonstration of GPIB communications **************************')
# open the connection with the specified instrument GPIB address
inst = Connection_Open_GPIB('12')
# send a command; returned tuple includes string response, if any, and 
# integer status byte
tReturn = Inst_Query_Command_GPIB(inst, "VER")
# decode and print the meaning of the status byte
Print_Status_Byte(tReturn[1])
# if response was generated print it
if (tReturn[0] != ''):
    print('Command response: ' + tReturn[0])
# close the connection
Connection_Close(inst)
print('\n')





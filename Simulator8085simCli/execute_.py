from lib2to3.pytree import Node
from Simulator8085simCli.Memory import RAM
from Simulator8085simCli.IO import IO
from Simulator8085simCli.ALU import ALU
from Simulator8085simCli.BUS import Bus
import Simulator8085simCli.CU
from Simulator8085simCli.CU import CU
import sys
import os
import time

def execute(string:str,*argv):
    
    MAX_ITERATIONS = 10000
    
    iteration_count = 0
    
    
    
    # python execute.py FILE.txt
    single_byte = [0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
                   0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
                   0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
                   0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,
                   0x0A, 0x1A, 0x02, 0x12, 0x76, 0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D, 0x04, 0x0C, 0x14,
                   0x1C, 0x24, 0x2C, 0x34, 0x3C, 0x03, 0x13, 0x23, 0x33, 0x0B, 0x1B, 0x2B, 0x3B, 0xC9, 0x80, 0x81,
                   0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0xA0, 0xA1,
                   0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xA8, 0xA9,
                   0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0x2F, 0xD8, 0xD0, 0xF0, 0xF8, 0xE8, 0xE0, 0xC8, 0xC0, 0xC5,
                   0xD5, 0xE5, 0xC1, 0xD1, 0xE1, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0x09, 0x19, 0x29,
                   0x39, 0x3F, 0xFE, 0x22, 0xF9, 0x37, 0x00, 0xE9, 0x17, 0x1F, 0x07, 0x0F, 0xEB, 0xE3]
    
    double_byte = [0x3E, 0x06, 0x0E, 0x16, 0x1E, 0x26, 0x2E, 0x36, 0xC6, 0xD6, 0xE6, 0xF6, 0xEE, 0xDB, 0xD3, 0xFE]
    
    triple_byte = [0x01, 0x11, 0x21, 0x2A, 0x31, 0x3A, 0x32, 0xDA, 0xD2, 0xF2, 0xFA, 0xEA, 0xE2, 0xCA, 0xC2, 0xC3, 0xC0,
                   0xCD, 0xDC, 0xD4, 0xF4, 0xFC, 0xEC, 0xE4, 0xCC, 0xC4]
    
    ram = RAM(0x2000,64)
    io = IO(0x01,0x02,0x03)
    alu = ALU()
    bus = Bus()
    bus.setMemoryPeripheral(ram,0x2000,0x6000)
    bus.setIOPeripheral(io,0x01)
    bus.setIOPeripheral(io,0x02)
    bus.setIOPeripheral(io,0x03)
    cu = CU(alu,bus)
    isError = False
    stop_detected = False
    start = 0x2000
    ERROR_RED = '\033[91m'
    def get_file():
        
        if(len(argv)>0):
            target_file = argv[0] #modif pour prendre la premire des options
            ext = os.path.splitext(target_file)[-1]
            if(ext!='.txt'):
                print('Invalid file')
            else:
                if(os.path.exists(target_file)):
                    #print('executing file '+target_file+'...'+'\n')
                    return target_file
                else:
                    os.chdir(os.path.dirname(target_file))
                    if(os.path.exists(os.path.basename(target_file))):
                        return target_file
                    else:
                        print('This file does not exists')
        else:
            print('No file specified')
        
    
    def get_options(opt):
        if opt in argv:
            if opt == '-start' and argv.index(opt)!=len(argv)-1:
                try:
                    start = int(argv[argv.index(opt)+1],16)
                    return start if start > 0x2000 and start < 0x6000 else None
                except ValueError:
                    return None
            elif opt == '-output' and argv.index(opt)!=len(argv)-1:
                file =  argv[argv.index(opt)+1]
                return file if os.path.splitext(file)[1] == '.txt' else None
            elif opt == '-read' and argv.index(opt)!=len(argv)-1:
                try:
                    loc = int(argv[argv.index(opt)+1],16)
                    if loc >= 0x2000 and loc <= 0x6000:
                        return loc
                    else:
                        raise ValueError('Cannot read this memory location')
                except ValueError:
                    return None
    
        
    
    def help():
        print('python execute.py [FILE] [OPTIONS]')
        print('Compile hexcodes from a text file and execute the instructions.')
        print("________________________________OPTIONS_________________________________")
        print('-reg Display registers in the output')
        print('-start [MEMLOC] Specify the starting memory location. Valid locations are from 2000 to 6000.')
        print('-port Display ports in the output')
        print('-output [FILENAME] Output result onto a text file instead of console')
        print('-read [MEM] Read a specific memory location')
        
    
            
    if '-help' in argv:
        help()
        sys.exit(0)
    
    # EXTRACTS CODE FROM TEXT FILE
    #file = get_file()
    #file = get_file()
    start = get_options('-start') if get_options('-start')!=None else start
    loc = start
    if(string==None):
        exit()
    
    
    #string = [i.split(';')[0].strip() for i in open(file,'r').read().split('\n')]
    string = string.split("\n")
    #print("string", string)
    
    # WRITING HEXCODES TO MEMORY
    #print('PC starting At '+hex(start))
    for item in string:
            try:
                if item.startswith('/') or item == '':
                    #comments
                    continue
    
                elif item.endswith(':'):
                    #address
                    item = item.strip(':')
                    loc = int(item,16)
                    if loc < 0x2000 or loc > 0x6000:
                        raise ValueError('Too large')
                    #print('PC changed to '+hex(loc))
                    continue
    
                elif int(item,16) > 0xff:
                    raise ValueError('Too large')
                elif int(item,16) == 0x76:
                    stop_detected = True
                
                #print(f'Writing {item} at {hex(loc)}')
                bus.WriteMemory(loc,int(item,16))
                loc += 1
            except ValueError:
                print(ERROR_RED+'Error at location '+hex(loc)+': invalid hex '+str(item)+'\033[0m')
                break 
                exit()
    
    if not stop_detected:
        print(ERROR_RED+'Error: Stop code not found.'+'\033[0m')
        sys.exit()
    
    #print('\033[92m'+'Done!'+'\033[0m')
                          
    cu.SetPC(start)
    
    
    
    
    
    
    
    # EXECUTES AND LOGS MESSAGES ONTO CONSOLE
    #print("________________________________INSTRUCTION LOG_________________________________")
    #print("PC starting At "+hex(cu.GetPC()))
    at_pc = "At PC "
    fetched_byte = " Fetched byte "
    
    tic = time.time()
    latence = 0
    while cu.running and iteration_count < MAX_ITERATIONS:
        try:
            hexcode = hex(bus.ReadMemory(cu.GetPC()))
            tic1 = time.time()
            if int(hexcode,16) == 0x00:
                tic2 = time.time()
                latence += tic2 - tic1
                tic1 = time.time()
                if cu.GetPC() < 0x6000:
                    tic2 = time.time()
                    latence += tic2 - tic1
                    cu.SetPC(cu.GetPC()+1)
                    continue
                else:
                    tic2 = time.time()
                    latence += tic2 - tic1
                    #print(ERROR_RED+'Error: Memory exhausted.'+'\033[0m')
                    isError = True
                    #tic = time.time()
                    break
                    #sys.exit()
            #print(at_pc+hex(cu.GetPC())+fetched_byte+str(bus.ReadMemory(cu.GetPC())) +"  "+"Hexcode: "+str(hexcode)+"   Opcode: "+str(cu.Logger()))
            tic1 = time.time()
            if bus.ReadMemory(cu.GetPC()) in double_byte:
                tic2 = time.time()
                latence += tic2 - tic1
                #print(at_pc+hex(cu.GetPC()+1)+fetched_byte+str(bus.ReadMemory(cu.GetPC()+1))+"   "+"Byte: "+hex(bus.ReadMemory(cu.GetPC()+1)))
            elif bus.ReadMemory(cu.GetPC()) in triple_byte:
                tic2 = time.time()
                latence += tic2 - tic1
                #print(at_pc+hex(cu.GetPC()+1)+fetched_byte+str(bus.ReadMemory(cu.GetPC()+1))+"   "+"Upper Byte: "+hex(bus.ReadMemory(cu.GetPC()+1)))
                #print(at_pc+hex(cu.GetPC()+2)+fetched_byte+str(bus.ReadMemory(cu.GetPC()+2))+"   "+"Lower Byte: "+hex(bus.ReadMemory(cu.GetPC()+2))) 
            elif bus.ReadMemory(cu.GetPC()) not in single_byte and bus.ReadMemory(cu.GetPC()) not in double_byte and bus.ReadMemory(cu.GetPC()) not in triple_byte:     
                tic2 = time.time()
                latence += tic2 - tic1
                #print("Invalid hexcode "+hex(bus.ReadMemory(cu.GetPC()))+" at "+hex(cu.GetPC()))
                isError = True
                break
            cu.FetchAndDecode()
    
            tic1 = time.time()
            if cu.jump:
                tic2 = time.time()
                latence += tic2 - tic1
                if not cu.stack:
                    #print("PC: "+hex(cu.GetPC())+"   jump count: "+str(cu.jump_count))
                    cu.jump = False
                else:
                    #print("PC: "+hex(cu.GetPC()))
                    cu.jump = False
            tic1 = time.time()
            if cu.stack:
                tic2 = time.time()
                latence += tic2 - tic1
                #print("SP: "+hex(alu.getRegister('SP')))
                cu.stack = False
            iteration_count += 1
        except Exception as e:
            #print(ERROR_RED+'Error at '+hex(cu.GetPC())+'\033[0m')
            break
            #sys.exit()
    
    toc = time.time()
    full_time = toc-tic
    #print("full time", full_time)
    #print("latence", latence)
    #print("counter", iteration_count)
    if iteration_count >= MAX_ITERATIONS:
        print(ERROR_RED+'Maximum iterations reached.'+'\033[0m')
    
    
    #print('\033[92m'+'Done!'+'\033[0m')
    
    
    # WRITES THE RESULTS INTO TEXT FILE OR CONSOLE
    if not isError:
        if '-output' not in argv:
            if '-reg' in argv:
                print("________________________________RESULTS_________________________________")
                print("Register content:")
                print("A: "+str(alu.getRegister('A')))
                print("B: "+str(alu.getRegister('B')))
                print("C: "+str(alu.getRegister('C')))
                print("D: "+str(alu.getRegister('D')))
                print("E: "+str(alu.getRegister('E')))
                print("H: "+str(alu.getRegister('H')))
                print("L: "+str(alu.getRegister('L')))
                print("F: "+str(alu.getRegister('FR')))
                print("PC: "+str(alu.getRegister('PC')))
                print("SP: "+str(alu.getRegister('SP')))
            if '-flag' in argv:
                print("Flag content:")
                print("F: "+str(alu.getRegister('F')))
            if '-port' in argv:
                print("Output Port:")
                print("Port A: "+str(bus.ReadIO(0x01)))
                print("Port B: "+str(bus.ReadIO(0x02)))
                print("Port C: "+str(bus.ReadIO(0x03)))
            if '-read' in argv:
                loc = get_options('-read')
                if loc != None:
                    print(f'Memory Location {hex(loc)}: {str(bus.ReadMemory(loc))}')
        else:
            file = get_options('-output')
            if file != None:
                print('Writing results to '+file)
                f = open(file,"w")
                if '-reg' in argv:
                    f.write("Register content:"+ '\n')
                    f.write("A: "+str(alu.getRegister('A'))+'\n')
                    f.write("B: "+str(alu.getRegister('B'))+'\n')
                    f.write("C: "+str(alu.getRegister('C'))+'\n')
                    f.write("D: "+str(alu.getRegister('D'))+'\n')
                    f.write("E: "+str(alu.getRegister('E'))+'\n')
                    f.write("H: "+str(alu.getRegister('H'))+'\n')
                    f.write("L: "+str(alu.getRegister('L'))+'\n')
                    f.write("F: "+str(alu.getRegister('FR'))+'\n')
                    f.write("PC: "+str(alu.getRegister('PC'))+'\n')
                    f.write("SP: "+str(alu.getRegister('SP'))+'\n')
                    f.write('\n')
                if '-flag' in argv:
                    f.write("Flag content:"+'\n')
                    f.write("F: "+str(alu.getRegister('F'))+'\n')
                    f.write('\n')
                if '-port' in argv:
                    f.write("Output Port:"+'\n')
                    f.write("Port A: "+str(bus.ReadIO(0x01))+'\n')
                    f.write("Port B: "+str(bus.ReadIO(0x02))+'\n')
                    f.write("Port C: "+str(bus.ReadIO(0x03))+'\n')
                    f.write('\n')
                if '-read' in argv:
                    loc = get_options('-read')
                    if loc != None:
                        f.write(f'Memory Location {hex(loc)}: {str(bus.ReadMemory(loc))}')
        #print('\x1b[6;30;42m' + 'Execution Completed!' + '\x1b[0m')
        
    return {"full_time": full_time,"latence": latence, "iteration_count":iteration_count }

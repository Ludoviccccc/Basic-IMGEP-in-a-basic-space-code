import numpy as np
def make_random_source_code(N = 100):
    instructions= ["NOP","LHLD", "SHLD"]
    list_instruction = np.random.randint(0,len(instructions), (N,))
    program =""
    for ch in list_instruction:
        program+= instructions[ch]
        if ch:
            arg = ""
            for i in range(2):
                #rand = np.random.randint(0,255)
                rand = 2024
                arg+=(rand<16)*"0" +hex(rand).split("0x")[1]
            program+=" "+ arg
        program+="\n"
    program+="HLT"
    return program


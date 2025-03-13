regOffset = [ 'B', 'C', 'D', 'E', 'H', 'L', 'M', 'A' ]
regPair = { 'H':'L', 'B':'C', 'D':'E', 'A':'F' }

# inr, dcr
mvi_map = { 'A':0x3E, 'B':0x06, 'C':0x0E, 'D':0x16, 'E':0x1E, 'H':0x26, 'L':0x2E, 'M':0x36 }
# inx, dcx, dad
lxi_map = { 'B':0x01, 'D':0x11, 'H':0x21, 'SP':0x31 }
# mov
mov_map = { 'A':0x78, 'B':0x40, 'C':0x48, 'D':0x50, 'E':0x58, 'H':0x60, 'L':0x68, 'M':0x70 }

push_map = { 'B':0xC5, 'D':0xD5, 'H':0xE5, 'PSW':0xF5 }
pop_map = { 'B':0xC1, 'D':0xD1, 'H':0xE1, 'PSW':0xF1 }
# rst
rst_map = [ 0xC7, 0xCF, 0xD7, 0xDF, 0xE7, 0xEF, 0xF7, 0xFF ]
#call, jump
cnd_map = { 'NZ':0xC2, 'Z':0xCA, 'NC':0xD2, 'C':0xDA, 'PO':0xE2, 'P':0xF2, 'M':0xFA }

stax_map = {'B':0x02, 'D':0x12}



ins_len2 = [ 'MVI', 'ADI', 'ACI', 'SUI', 'SBI', 'ANI', 'XRI', 'ORI', 'CPI',
             'OUT', 'IN' ]

ins_len3 = [ 'LXI', 'JMP', 'JNZ', 'JZ', 'JNC', 'JC', 'JPO', 'JPE', 'JP', 'JM',
                    'CALL', 'CNZ', 'CZ', 'CNC', 'CC', 'CPO', 'CPE', 'CP', 'CM',
                    'LHLD', 'LDA', 'STA', 'SHLD']

imm_opcodes = { 'ADI':0xC6, 'ACI':0xCE, 'SUI':0xD6, 'SBI':0xDE, 'ANI':0xE6, 'XRI':0xEE, 'ORI':0xF6, 'CPI':0xFE, 'OUT':0xD3, 'IN':0xDB }
addr_opcodes = { 'LDA': 0x3A, 'STA':0x32, 'JMP':0xC3, 'CALL':0xCD, 'LHLD':0x2A, 'SHLD':0x22 }

singlebytereg_opcodes = { 'ADD':0x80, 'ADC':0x88, 'SUB':0x90, 'SBB':0x98, 'ANA':0xA0, 'XRA':0xA8, 'ORA':0xB0, 'CMP':0xB8 }

singlebyte_opcodes = { 'RET':0xC9, 'HLT':0x76, 'RLC':0x07, 'RRC':0x0F, 'RAL':0x17, 'RAR':0x1F, 'CMA':0x2E, 'DAA':0x27, 'CMC':0x3F,
                        'STC':0x37, 'EI':0xFB, 'DI':0xFB, 'NOP':0x00, 'XTHL':0xE3, 'SPHL':0xF9, 'SIM':0x30, 'RIM':0x20, 'XCHG':0xEB }

misc_opcodes = [ 'MOV', 'INX', 'DCX', 'INR', 'DCR', 'PUSH', 'POP', 'DAD', 'RST' ]


def compiler(codestr):
    codestr = codestr.upper()
    words = codestr.split()
    res = []
    if words[0] in singlebyte_opcodes:          # hlt, ret
        res.append(singlebyte_opcodes[words[0]])
    elif words[0] in singlebytereg_opcodes:     # add b
        res.append(singlebytereg_opcodes[words[0]]+regOffset.index(words[1]))
    elif words[0] in ins_len2:
        if words[0] in imm_opcodes:             # adi 0a
            res.append(imm_opcodes[words[0]])
            res.append(int(words[1],16))
        else:                                   # mvi a ff
            res.append(mvi_map[words[1]])
            res.append(int(words[2],16))
    elif words[0] in misc_opcodes:
        if words[0]=='MOV':                     # mov a,b
            res.append(mov_map[words[1][0]]+regOffset.index(words[1][2]))
        elif words[0]=='INX':                   # inx d
            res.append(lxi_map[words[1]]+2)
        elif words[0]=='DCX':                   # dcx b
            res.append(lxi_map[words[1]]+0xA)
        elif words[0]=='INR':                   # inr a
            res.append(mvi_map[words[1]]-2)
        elif words[0]=='DCR':                   # dcr a
            res.append(mvi_map[words[1]]-1)
        elif words[0]=='PUSH':                  # push b
            res.append(push_map[words[1]])
        elif words[0]=='POP':                   # pop psw
            res.append(pop_map[words[1]])
        elif words[0]=='DAD':                   # dad sp or dad b
            res.append(lxi_map[words[1]]+8)
        elif words[0]=='RST':                   # rst 5
            res.append(rst_map[int(words[1])])
    elif words[0] in ins_len3:
        if words[0] in addr_opcodes:            # jmp 2050, lhdl 2050, 
            res.append(addr_opcodes[words[0]])
            res.append(int(words[1][2]+words[1][3],16))
            res.append(int(words[1][0]+words[1][1],16))
        else:
            if words[0][0]=='J':                # jnz 2050 (jc, jm, jnc ...)
                res.append(cnd_map[words[0][1:]])
                res.append(int(words[1][2]+words[1][3],16))
                res.append(int(words[1][0]+words[1][1],16))
            elif words[0][0]=='C':              # cnz 2050
                res.append(cnd_map[words[0][1:]]+2)
                res.append(int(words[1][2]+words[1][3],16))
                res.append(int(words[1][0]+words[1][1],16))
            elif words[0]=='LXI':               # lxi d 40ff
                res.append(lxi_map[words[1]])
                res.append(int(words[2][2]+words[2][3],16))
                res.append(int(words[2][0]+words[2][1],16))
    elif words[0]=='STAX':                      # stax b
        res.append(stax_map[words[1]])
    elif words[0]=='LDAX':
        res.append(stax_map[words[1]]+8)
    elif words[0][0]=='R':
        res.append(cnd_map[words[0][1:]]-2)

    return res





def compile_program(program:str)->str:
    res = []
    output =""
    for i in program.strip().split('\n'):
        res.extend([hex(x) for x in compiler(i)])
        for ch in [hex(x) for x in compiler(i)]:
            output+=ch[2:]+"\n"
    return output


# program = """MVI A 0x0
# MVI B 0x1
# ADD B
# JMP 0020
# HLT"""

#program = """SHLD 2024
#NOP
#LHLD 2024
#SHLD 2024
#NOP
#LHLD 2024
#HLT"""
#
#program = """SHLD 2024
#NOP
#SHLD 2025
#NOP
#LHLD 2024
#HLT"""
#print(program)
#print(type(program))
#
#compiled=compile_program(program)
#
##print(compiled)
#
#with open('program.txt','w') as f:
#    for i in compiled:
#        f.write(i[2:]+'\n')
#
#
#


instruction_set = ['MVI','LXI','MOV','LDA','LDAX','STA','STAX','ADD','ADI','SUB','SUI','ANA','ANI','ORA',
                   'ORI','XRA','XRI','INR','DCR','INX','DCX','CMA','JMP','JC', 'JNC','JP','JM','JPE','JPO',
                   'JZ','JNZ','CALL','RET', 'CC', 'CNC','CP','CM','CPE','CPO','CZ','CNZ','RC', 'RNC','RP',
                   'RM','RPE','RPO','RZ','RNZ','IN','OUT']

mvi_map =  { 0x3E:'A', 0x06:'B', 0x0E:'C', 0x16:'D', 0x1E:'E', 0x26:'H', 0x2E:'L', 0x36:'M' }
lxi_map =  { 0x01:'B', 0x11:'D', 0x21:'H', 0x31:'SP' }
ldax_map = { 0x0A:'B', 0x1A:'D'}
stax_map = { 0x02:'B', 0x12:'D'}
push_map = { 0xC5:'B', 0xD5:'D', 0xE5:'H', 0xF5:'PSW'}
pop_map =  { 0xC1:'B', 0xD1:'D', 0xE1:'H', 0xF1:'PSW'}
mov_map =  { 0x40:['B','B'], 0x41:['B','C'], 0x42:['B','D'], 0x43:['B','E'], 0x44:['B','H'], 0x45:['B','L'],
             0x46:['B','M'], 0x47:['B','A'], 0x48:['C','B'], 0x49:['C','C'], 0x4A:['C','D'], 0x4B:['C','E'],
             0x4C:['C','H'], 0x4D:['C','L'], 0x4E:['C','M'], 0x4F:['C','A'], 0x50:['D','B'], 0x51:['D','C'],
             0x52:['D','D'], 0x53:['D','E'], 0x54:['D','H'], 0x55:['D','L'], 0x56:['D','M'], 0x57:['D','A'],
             0x58:['E','B'], 0x59:['E','C'], 0x5A:['E','D'], 0x5B:['E','E'], 0x5C:['E','H'], 0x5D:['E','L'],
             0x5E:['E','M'], 0x5F:['E','A'], 0x60:['H','B'], 0x61:['H','C'], 0x62:['H','D'], 0x63:['H','E'],
             0x64:['H','H'], 0x65:['H','L'], 0x66:['H','M'], 0x67:['H','A'], 0x68:['L','B'], 0x69:['L','C'],
             0x6A:['L','D'], 0x6B:['L','E'], 0x6C:['L','H'], 0x6D:['L','L'], 0x6E:['L','M'], 0x6F:['L','A'],
             0x70:['M','B'], 0x71:['M','C'], 0x72:['M','D'], 0x73:['M','E'], 0x74:['M','H'], 0x75:['M','L'],
             0x77:['M','A'], 0x78:['A','B'], 0x79:['A','C'], 0x7A:['A','D'], 0x7B:['A','E'],
             0x7C:['A','H'], 0x7D:['A','L'], 0x7E:['A','M'], 0x7F:['A','A']}

add_map = { 0x80:'B', 0x81:'C', 0x82:'D', 0x83:'E', 0x84:'H', 0x85:'L', 0x86:'M', 0x87:'A'}
sub_map = { 0x90:'B', 0x91:'C', 0x92:'D', 0x93:'E', 0x94:'H', 0x95:'L', 0x96:'M', 0x97:'A'}
and_map = { 0xA0:'B', 0xA1:'C', 0xA2:'D', 0xA3:'E', 0xA4:'H', 0xA5:'L', 0xA6:'M', 0xA7:'A'}
or_map =  { 0xB0:'B', 0xB1:'C', 0xB2:'D', 0xB3:'E', 0xB4:'H', 0xB5:'L', 0xB6:'M', 0xB7:'A'}
xr_map =  { 0xA8:'B', 0xA9:'C', 0xAA:'D', 0xAB:'E', 0xAC:'H', 0xAD:'L', 0xAE:'M', 0xAF:'A'}
inr_map = { 0x04:'B', 0x0C:'C', 0x14:'D', 0x1C:'E', 0x24:'H', 0x2C:'L', 0x34:'M', 0x3C:'A'}
dcr_map = { 0x05:'B', 0x0D:'C', 0x15:'D', 0x1D:'E', 0x25:'H', 0x2D:'L', 0x35:'M', 0x3D:'A'}
cmp_map = { 0xB8:'B', 0xB9:'C', 0xBA:'D', 0xBB:'E', 0xBC:'H', 0xBD:'L', 0xBE:'M', 0xBF:'A'}
dad_map = { 0x09:'B', 0x19:'D', 0x29:'H', 0x39:'SP'}
inx_map = { 0x03:'B', 0x13:'D', 0x23:'H', 0x33:'SP'}
dcx_map = { 0x0B:'B', 0x1B:'D', 0x2B:'H', 0x3B:'SP'}
jump_map ={ 0xDA:'JC', 0xD2:'JNC', 0xF2:'JP', 0xFA:'JM', 0xEA:'JPE', 0xE2:'JPO', 0xCA:'JZ', 0xC2:'JNZ'}
call_map ={ 0xDC:'CC', 0xD4:'CNC', 0xF4:'CP', 0xFC:'CM', 0xEC:'CPE', 0xE4:'CPO', 0xCC:'CZ', 0xC4:'CNZ'}
ret_map = { 0xD8:'RC', 0xD0:'RNC', 0xF0:'RP', 0xF8:'RM', 0xE8:'RPE', 0xE0:'RPO', 0xC8:'RZ', 0xC0:'RNZ'}
other_map={ 0x76:'hlt', 0x3A:'lda', 0x32:'sta', 0xC6:'adi', 0xD6:'sui', 0xE6:'ani', 0xF6:'ori', 0xEE:'xri',
            0x2F:'cma', 0xC3:'jmp', 0xCD:'call', 0xC9:'ret', 0xD3:'out', 0xDB:'in', 0x3F:'cmc', 0xFE:'cpi',
            0x22:'shld',0xF9:'sphl',0x37:'stc', 0x00:'nop', 0xE9:'pchl', 0x17:'ral', 0x1F:'rar',0x07:'rlc',
            0x0F:'rrc', 0xEB:'xchg',0xE3:'xthl'}
            
        
        
        
        

class CU:

    def __init__(self,alu,bus):
        self.alu = alu
        self.bus = bus
        self.running = True
        self.jump = False
        self.stack = False
        self.jump_count = 0
        

    def GetPC(self):
        return self.alu.registers['PC']

    def GetSP(self):
        return self.alu.registers['SP']

    def SetPC(self, addr):
        self.alu.setRegister('PC',addr)

    def SetSP(self,addr):
        self.alu.setRegister('SP',addr)

    def Fetch(self):
        pc = self.GetPC()
        self.SetPC(pc+1)
        byte = self.bus.ReadMemory(pc)
        return byte

    def FetchAndDecode(self):
        opcode = self.Fetch()
        if opcode in mvi_map.keys():
            self.Mvi(opcode)
        elif opcode in lxi_map.keys():
            self.Lxi(opcode)
        elif opcode in mov_map.keys():
            self.Mov(opcode)
        elif opcode == 0x3A:
            self.Lda()
        elif opcode in ldax_map.keys():
            self.Ldax(opcode)
        elif opcode in push_map.keys():
            self.Push(opcode)
        elif opcode in pop_map.keys():
            self.Pop(opcode)
        elif opcode == 0x32:
            self.Sta()
        elif opcode in stax_map.keys():
            self.Stax(opcode)
        elif opcode == 0xC6:
            self.Adi()
        elif opcode in add_map.keys():
            self.Add(opcode)
        elif opcode == 0xD6:
            self.Sui()
        elif opcode in sub_map.keys():
            self.Sub(opcode)
        elif opcode in and_map.keys():
            self.Ana(opcode)
        elif opcode == 0xE6:
            self.Ani()
        elif opcode in or_map.keys():
            self.Ora(opcode)
        elif opcode == 0xF6:
            self.Ori()
        elif opcode == 0xEE:
            self.Xri()
        elif opcode in xr_map.keys():
            self.Xra(opcode)
        elif opcode in inr_map.keys():
            self.Inr(opcode)
        elif opcode in dcr_map.keys():
            self.Dcr(opcode)
        elif opcode in inx_map.keys():
            self.Inx(opcode)
        elif opcode in dcx_map.keys():
            self.Dcx(opcode)
        elif opcode == 0x2F:
            self.Cma()
        elif opcode == 0x3F:
            self.Cmc()
        elif opcode == 0xC3:
            self.Jmp()
        elif opcode in cmp_map.keys():
            self.Cmp(opcode)
        elif opcode == 0xFE:
            self.Cpi()
        elif opcode in dad_map.keys():
            self.Dad(opcode)
        elif opcode == 0x2A:
            self.Lhld()
        elif opcode == 0x22:
            self.Shld()
        elif opcode == 0xF9:
            self.Sphl()
        elif opcode == 0x37:
            self.Stc()
        elif opcode == 0x00:
            self.Nop()
        elif opcode == 0xE9:
            self.Pchl()
        elif opcode == 0x17:
            self.Ral()
        elif opcode == 0x1F:
            self.Rar()
        elif opcode == 0x07:
            self.Rlc()
        elif opcode == 0x0F:
            self.Rrc()
        elif opcode == 0xEB:
            self.Xchg()
        elif opcode == 0xE3:
            self.Xthl()
        elif opcode in jump_map.keys():
            self.Jmp_Cond(opcode)
        elif opcode == 0xcd:
            self.Call()
        elif opcode == 0xc9:
            self.Ret()
        elif opcode in call_map.keys():
            self.Call_Cond(opcode)
        elif opcode in ret_map.keys():
            self.Ret_Cond(opcode)
        elif opcode == 0xD3:
            self.Out()
        elif opcode == 0xDB:
            self.In()
        elif opcode == 0x76:
            self.running = False
        else:
            self.running = False
            print("Invalid opcode "+ hex(opcode) +" at "+hex(self.GetPC()))

    def Mvi(self,opcode):
        reg = mvi_map[opcode]
        byte = self.Fetch()
        if reg == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.bus.WriteMemory(addr_byte,byte)
        else:
            self.alu.setRegister(reg,byte)

    def Lxi(self,opcode):
        regpair = lxi_map[opcode]
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        if regpair == 'SP':
            byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            self.alu.setRegister('SP',int(byte,16))
        else:
            self.alu.setRegister(regpair,byte_h)
            self.alu.setRegister(self.alu.regPair[regpair],byte_l)
        
    def Mov(self,opcode):
        reg_d = mov_map[opcode][0]
        reg_s = mov_map[opcode][1]
        if reg_d == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.bus.WriteMemory(addr_byte,self.alu.getRegister(reg_s))
        elif reg_s == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.alu.setRegister(reg_d,self.bus.ReadMemory(addr_byte))
        else:
            self.alu.Mov(reg_d,reg_s)

    def Lda(self):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        self.alu.setRegister('A',self.bus.ReadMemory(int(addr_byte,16)))

    def Ldax(self,opcode):
        addr_byte = self.alu.getRegisterPair(ldax_map[opcode])
        self.alu.setRegister('A',self.bus.ReadMemory(addr_byte))

    def Push(self,opcode):
        stk_ptr = self.alu.getRegister('SP')
        stk_ptr -= 1
        reg_pair = self.alu.regPair[push_map[opcode]]
        self.bus.WriteMemory(stk_ptr,self.alu.getRegister(push_map[opcode]))
        self.alu.setRegister('SP',stk_ptr)
        stk_ptr -= 1
        self.bus.WriteMemory(stk_ptr,self.alu.getRegister(reg_pair))
        self.alu.setRegister('SP',stk_ptr)

    def Pop(self,opcode):
        stk_ptr = self.alu.getRegister('SP')
        reg_pair = self.alu.regPair[pop_map[opcode]]
        self.alu.setRegister(reg_pair,self.bus.ReadMemory(stk_ptr))
        if reg_pair == 'FR':
            self.alu.setFlagsByRegister()
        stk_ptr += 1
        self.alu.setRegister('SP',stk_ptr)
        self.alu.setRegister(pop_map[opcode],self.bus.ReadMemory(stk_ptr))
        stk_ptr += 1
        self.alu.setRegister('SP',stk_ptr)

    def Sta(self):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        byte_addr = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        self.bus.WriteMemory(int(byte_addr,16),self.alu.getRegister('A'))

    def Stax(self,opcode):
        byte_addr = self.alu.getRegisterPair(stax_map[opcode])
        self.bus.WriteMemory(byte_addr,self.alu.getRegister('A'))
        
    def Adi(self):
        byte = self.Fetch()
        self.alu.Adi(byte)

    def Add(self,opcode):
        if add_map[opcode] == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.alu.Adi(self.bus.ReadMemory(addr_byte))
        else:
            self.alu.Add(add_map[opcode])

    def Sui(self):
        byte = self.Fetch()
        self.alu.Sui(byte)

    def Sub(self,opcode):
        if sub_map[opcode] == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.alu.Sui(self.bus.ReadMemory(addr_byte))
        else:
            self.alu.Sub(sub_map[opcode])

    def Ana(self,opcode):
        if and_map[opcode] == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.alu.Ani(self.bus.ReadMemory(addr_byte))
        else:
            self.alu.Ana(and_map[opcode])

    def Ani(self):
        byte = self.Fetch()
        self.alu.Ani(byte)

    def Ora(self,opcode):
        if or_map[opcode] == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.alu.Ori(self.bus.ReadMemory(addr_byte))
        else:
            self.alu.Ora(or_map[opcode])

    def Ori(self):
        byte = self.Fetch()
        self.alu.Ori(byte)

    def Xra(self,opcode):
        if xr_map[opcode] == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            self.alu.Xri(self.bus.ReadMemory(addr_byte))
        else:
            self.alu.Xra(xr_map[opcode])

    def Xri(self):
        byte = self.Fetch()
        self.alu.Xri(byte)

    def Cma(self):
        self.alu.Cma()

    def Cmc(self):
        self.alu.Cmc()

    def Cmp(self,opcode):
        if cmp_map[opcode] == 'M':
            self.alu.Cpi(self.bus.ReadMemory(self.alu.getRegisterPair('H')))
        else:
            self.alu.Cmp(cmp_map[opcode])

    def Cpi(self):
        byte = self.Fetch()
        self.alu.Cpi(byte)

    def Inr(self,opcode):
        reg = inr_map[opcode]
        if reg == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            byte = self.bus.ReadMemory(addr_byte)
            self.bus.WriteMemory(addr_byte,byte+1)
        else:
            self.alu.Inr(reg)
    
    def Dcr(self,opcode):
        reg = dcr_map[opcode]
        if reg == 'M':
            addr_byte = self.alu.getRegisterPair('H')
            byte = self.bus.ReadMemory(addr_byte)
            self.bus.WriteMemory(addr_byte,byte-1)
        else:
            self.alu.Dcr(reg)

    def Inx(self,opcode):
        regpair = inx_map[opcode]
        self.alu.Inx(regpair)

    def Dcx(self,opcode):
        regpair = dcx_map[opcode]
        self.alu.Dcx(regpair)

    def Dad(self,opcode):
        self.alu.Dad(dad_map[opcode])

    def Lhld(self):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        self.alu.setRegister('L',self.bus.ReadMemory(int(addr_byte,16)))
        self.alu.setRegister('H',self.bus.ReadMemory(int(addr_byte,16)+1))

    def Shld(self):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        self.bus.WriteMemory(int(addr_byte,16),self.alu.getRegister('L'))
        self.bus.WriteMemory(int(addr_byte,16)+1,self.alu.getRegister('H'))

    def Nop(self):
        return

    def Stc(self):
        self.alu.registers['F']['CY'] = 1

    def Xchg(self):
        temp = self.alu.getRegisterPair('H') 
        self.alu.setRegisterPair('H',self.alu.getRegisterPair('D'))
        self.alu.setRegisterPair('D',temp)

    def Xthl(self):
        temp = self.alu.getRegister('L')
        self.alu.setRegister('L',self.bus.ReadMemory(self.GetSP()))
        self.bus.WriteMemory(self.GetSP(),temp)
        temp = self.alu.getRegister('H')
        self.alu.setRegister('H',self.bus.ReadMemory(self.GetSP()+1))
        self.bus.WriteMemory(self.GetSP()+1,temp)
        

    def Pchl(self):
        self.SetPC(self.alu.getRegisterPair('H'))

    def Sphl(self):
        self.SetSP(self.alu.getRegisterPair('H'))

    def Ral(self):
        self.alu.Ral()

    def Rar(self):
        self.alu.Rar()

    def Rlc(self):
        self.alu.Rlc()

    def Rrc(self):
        self.alu.Rrc()

    def Jmp(self):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        self.jump = True
        self.SetPC(int(addr_byte,16))
        self.jump_count += 1
    
    def Jmp_Cond(self,opcode):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        if jump_map[opcode] == 'JC':
            if self.alu.registers['F']['CY']:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JNC':
            if self.alu.registers['F']['CY'] == 0:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JP':
            if self.alu.registers['F']['S'] == 0:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JM':
            if self.alu.registers['F']['S']:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JPE':
            if self.alu.registers['F']['P']:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JPO':
            if self.alu.registers['F']['P'] == 0:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JZ':
            if self.alu.registers['F']['Z']:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return
        elif jump_map[opcode] == 'JNZ':
            if self.alu.registers['F']['Z'] == 0:
                self.jump = True
                self.SetPC(int(addr_byte,16))
                self.jump_count += 1
            else:
                return


    def Call(self):
        stk_ptr = self.alu.getRegister('SP')
        stk_ptr -= 1
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        pc = hex(self.GetPC()).replace('0x','').zfill(4)
        self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
        self.alu.setRegister('SP',stk_ptr)
        stk_ptr -= 1
        self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
        self.alu.setRegister('SP',stk_ptr)
        self.SetPC(int(addr_byte,16))
        self.jump = True
        self.stack = True
        
    def Ret(self):
        stk_ptr = self.alu.getRegister('SP')
        byte_l = self.bus.ReadMemory(stk_ptr)
        stk_ptr += 1
        self.alu.setRegister('SP',stk_ptr)
        byte_h = self.bus.ReadMemory(stk_ptr)
        stk_ptr += 1
        self.alu.setRegister('SP',stk_ptr)
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        self.SetPC(int(addr_byte,16))
        self.jump = True
        self.stack = True
        
    def Call_Cond(self,opcode):
        byte_h = self.Fetch()
        byte_l = self.Fetch()
        addr_byte = hex(byte_h).replace('0x','').zfill(2) + hex(byte_l).replace('0x','').zfill(2)
        if call_map[opcode] == 'CC':
            if self.alu.registers['F']['CY']:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CNC':
            if self.alu.registers['F']['CY']==0:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CP':
            if self.alu.registers['F']['S']==0:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CM':
            if self.alu.registers['F']['S']:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CPE':
            if self.alu.registers['F']['P']:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CPO':
            if self.alu.registers['F']['P']==0:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CZ':
            if self.alu.registers['F']['Z']:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return
        elif call_map[opcode] == 'CNZ':
            if self.alu.registers['F']['Z']==0:
                stk_ptr = self.alu.getRegister('SP')
                stk_ptr -= 1
                pc = hex(self.GetPC()).replace('0x','').zfill(4)
                self.bus.WriteMemory(stk_ptr,int(pc[:2],16))
                self.alu.setRegister('SP',stk_ptr)
                stk_ptr -= 1
                self.bus.WriteMemory(stk_ptr,int(pc[2:],16))
                self.alu.setRegister('SP',stk_ptr)
                self.SetPC(int(addr_byte,16))
                self.jump = True
                self.stack = True
            else:
                return

    def Ret_Cond(self,opcode):
        if ret_map[opcode] == 'RC':
            if self.alu.registers['F']['CY']:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RNC':
            if self.alu.registers['F']['CY']==0:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RP':
            if self.alu.registers['F']['S']==0:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RM':
            if self.alu.registers['F']['S']:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RPE':
            if self.alu.registers['F']['P']:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RPO':
            if self.alu.registers['F']['P']==0:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RZ':
            if self.alu.registers['F']['Z']:
                self.Ret()
            else:
                return
        elif ret_map[opcode] == 'RNZ':
            if self.alu.registers['F']['Z']==0:
                self.Ret()
            else:
                return

    def In(self):
        byte = self.Fetch()
        self.alu.setRegister('A',self.bus.ReadIO(byte))

    def Out(self):
        byte = self.Fetch()
        self.bus.WriteIO(byte,self.alu.getRegister('A'))



    def Logger(self):
        opcode = self.bus.ReadMemory(self.GetPC())
        if opcode == 0x76:
            return "hlt"
        elif opcode in mvi_map.keys():
            if mvi_map[opcode] == 'M':
                return "mvi M, "+hex(self.bus.ReadMemory(self.GetPC()+1))
            else:
                return "mvi "+mvi_map[opcode]+", "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode in lxi_map.keys():
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            if lxi_map[opcode] == 'SP':
                self.stack = True
            return "lxi "+lxi_map[opcode]+", "+addr_byte
        elif opcode in mov_map.keys():
            return "mov "+mov_map[opcode][0]+", "+mov_map[opcode][1]
        elif opcode == 0x3A:
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            return "lda "+addr_byte
        elif opcode in ldax_map.keys():
            return "ldax "+ldax_map[opcode]
        elif opcode in push_map.keys():
            self.stack = True
            return "push "+push_map[opcode]
        elif opcode in pop_map.keys():
            self.stack = True
            return "pop "+pop_map[opcode]
        elif opcode == 0x32:
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            return "sta "+addr_byte
        elif opcode in stax_map.keys():
            return "stax "+stax_map[opcode]
        elif opcode == 0xC6:
            return "adi "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode in add_map.keys():
            return "add "+add_map[opcode]
        elif opcode == 0xD6:
            return "sui "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode in sub_map.keys():
            return "sub "+sub_map[opcode]
        elif opcode in and_map.keys():
            return "ana "+and_map[opcode]
        elif opcode == 0xE6:
            return "ani "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode in or_map.keys():
            return "ora "+or_map[opcode]
        elif opcode == 0xF6:
            return "ori "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode == 0xEE:
            return "xri "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode in xr_map.keys():
            return "xra "+xr_map[opcode]
        elif opcode == 0x2F:
            return "cma "
        elif opcode == 0xC3:
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            return "jmp "+ addr_byte
        elif opcode in inr_map.keys():
            return "inr "+inr_map[opcode]
        elif opcode in dcr_map.keys():
            return "dcr "+dcr_map[opcode]
        elif opcode in inx_map.keys():
            return "inx "+inx_map[opcode]
        elif opcode in dcx_map.keys():
            return "dcx "+dcx_map[opcode]

        elif opcode in jump_map:
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            if jump_map[opcode] == 'JC':
                return "jc "+addr_byte
            elif jump_map[opcode] == 'JNC':
                return "jnc "+addr_byte
            elif jump_map[opcode] == 'JP':
                return "jp "+addr_byte
            elif jump_map[opcode] == 'JM':
                return "jm "+addr_byte
            elif jump_map[opcode] == 'JPE': 
                return "jpe "+addr_byte  
            elif jump_map[opcode] == 'JPO':  
                return "jpo "+addr_byte
            elif jump_map[opcode] == 'JZ':
                return "jz "+addr_byte
            elif jump_map[opcode] == 'JNZ':
                return "jnz "+addr_byte

        elif opcode == 0xcd:
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            return "call "+addr_byte

        elif opcode in call_map.keys():
            byte_h = self.bus.ReadMemory(self.GetPC()+1)
            byte_l = self.bus.ReadMemory(self.GetPC()+2)
            addr_byte = hex(byte_h).zfill(2) + hex(byte_l).replace('0x','').zfill(2)
            if call_map[opcode] == 'CC':
                return "cc "+addr_byte
            elif call_map[opcode] == 'CNC':
                return "cnc "+addr_byte
            elif call_map[opcode] == 'CP':
                return "cp "+addr_byte
            elif call_map[opcode] == 'CM':
                return "cm "+addr_byte
            elif call_map[opcode] == 'CPE':
                return "cpe "+addr_byte
            elif call_map[opcode] == 'CPO':
                return "cpo "+addr_byte
            elif call_map[opcode] == 'CZ':
                return "cz "+addr_byte
            elif call_map[opcode] == 'CNZ':
                return "cnz "+addr_byte

        elif opcode == 0xc9:
            return "ret "

        elif opcode in ret_map.keys():
            if ret_map[opcode] == 'RC':
                return "rc "
            elif ret_map[opcode] == 'RNC':
                return "rnc "
            elif ret_map[opcode] == 'RP':
                return "rp "
            elif ret_map[opcode] == 'RM':
                return "rm "
            elif ret_map[opcode] == 'RPE':
                return "rc "
            elif ret_map[opcode] == 'RPO':
                return "rpo "
            elif ret_map[opcode] == 'RZ':
                return "rz "
            elif ret_map[opcode] == 'RNZ':
                return "rnz "

        elif opcode == 0xD3:
            return "out "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode == 0xDB:
            return "in "+hex(self.bus.ReadMemory(self.GetPC()+1))
            
        elif opcode in cmp_map.keys():
            return "cmp "+cmp_map[opcode]
        elif opcode == 0xFE:
            return "cpi "+hex(self.bus.ReadMemory(self.GetPC()+1))
        elif opcode in dad_map.keys():
            return "dad "+dad_map.keys()
        elif opcode == 0x2A:
            return "lhld "
        elif opcode == 0x22:
            return "shld "
        elif opcode == 0xF9:
            return "sphl "
        elif opcode == 0x37:
            return "stc "
        elif opcode == 0x00:
            return "nop "
        elif opcode == 0xE9:
            return "pchl "
        elif opcode == 0x17:
            return "ral "
        elif opcode == 0x1F:
            return "rar "
        elif opcode == 0x07:
            return "rlc "
        elif opcode == 0x0F:
            return "rrc "
        elif opcode == 0xEB:
            return "xchg "
        elif opcode == 0xE3:
            return "xthl "
        elif opcode == 0x3F:
            return "cmc "

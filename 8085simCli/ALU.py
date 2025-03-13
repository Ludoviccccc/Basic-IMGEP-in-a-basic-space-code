

class ALU:

    def __init__(self):
        self.registers = { 'A':0,'B':0,'C':0,'D':0,'E':0,'H':0,'L':0,'F':{'CY':0,'P':0,'AC':0,'Z':0,'S':0},'FR':0,'PC':0, 'SP':0xFF }
        self.regPair = { 'H':'L', 'B':'C', 'D':'E','A':'FR'}

    def getRegister(self,reg):
        return self.registers[reg]

    def getRegisterPair(self,reg):
        reg2 = self.regPair[reg]
        val = hex(self.registers[reg]).replace('0x','').zfill(2) + hex(self.registers[reg2]).replace('0x','').zfill(2)
        return int(val,16)

    def setRegister(self,reg,val):
        self.registers[reg]=val
        self.checkAll()
        self.setFlagRegister()

    def Mov(self,reg1,reg2):
        self.registers[reg1]=self.registers[reg2]
        self.checkAll()
        self.setFlagRegister()

    def setRegisterPair(self,reg,val):
        reg2 = self.regPair[reg]
        val = hex(val).replace('0x','').zfill(4)
        self.registers[reg2] = int(val[2:],16)
        self.registers[reg] = int(val[:2],16)
        self.checkAll()
        self.setFlagRegister()

    def setFlagRegister(self):
        self.registers['FR'] = self.registers['F']['CY'] + (4*self.registers['F']['P']) + (16*self.registers['F']['AC']) + (64*self.registers['F']['Z']) + (128*self.registers['F']['S'])
        
    def setFlagsByRegister(self):
        val = self.registers['FR']
        val = bin(val).replace('0b','').zfill(8)
        self.registers['F']['CY'] = int(val[7],2)
        self.registers['F']['P'] = int(val[5],2)
        self.registers['F']['AC'] = int(val[3],2)
        self.registers['F']['Z'] = int(val[1],2)
        self.registers['F']['S'] = int(val[0],2)

    def setCarry(self):
        self.registers['F']['CY']=1

    def setAuxCarry(self):
        self.registers['F']['AC']=1
        
    def setZero(self):
        self.registers['F']['Z']=1
        
        
    def setParity(self,ToSet):
        if ToSet:
            self.registers['F']['P']=1
        else:
            self.registers['F']['P']=0

    def setSign(self,ToSet):
        if ToSet:
            self.registers['F']['S']=1
        else:
            self.registers['F']['S']=0

    def checkCarry(self,reg):
        val = self.registers[reg]
        if val > 0xff:
            self.setCarry()
        
    def checkBorrow(self,reg):
        val = self.registers[reg]
        if val < 0x00:
            self.setCarry()
        
    def checkZero(self):
        val = self.registers['A']
        if val == 0x00:
            self.setZero()
        else:
            self.registers['F']['Z']=0
        

    def checkParity(self):
        val = self.registers['A']
        if val < 0x00:
            val=val*-1
        p = 0
        while val:
            p = ~p
            val = val & (val-1)
        self.setParity(p==0)

    def checkSign(self):
        val = self.registers['A']
        self.setSign(val>>31==-1)

    def checkAuxCarry(self,c1,c2):
        if c1 + c2 > 15:
            self.setAuxCarry()

    def checkAuxBorrow(self,c1,c2):
        if c1 - c2 < 0:
            self.setAuxCarry()
        
    def checkAll(self):
        self.checkZero()
        self.checkParity()
        self.checkSign()

    def lowerNibble(self,val):
        val = hex(val)
        c = val[len(val)-1]
        return int(c,16)
        
    def Add(self,reg):
        res = self.registers['A'] + self.registers[reg]
        self.checkAuxCarry(self.lowerNibble(self.registers['A']),self.lowerNibble(self.registers[reg]))
        self.registers['A'] = res
        self.checkCarry('A')
        if self.registers['F']['CY']:
            res = hex(res).replace('0x','')
            self.registers['A'] = int(res[1:],16)
        self.checkAll()
        self.setFlagRegister()

    def Adi(self,val):
        res = self.registers['A'] + val
        self.checkAuxCarry(self.lowerNibble(self.registers['A']),self.lowerNibble(val))
        self.registers['A'] = res
        self.checkCarry('A')
        if self.registers['F']['CY']:
            res = hex(res).replace('0x','')
            self.registers['A'] = int(res[1:],16)
        self.checkAll()
        self.setFlagRegister()

    def Sub(self,reg):
        res = self.registers['A'] + (~self.registers[reg]+1)
        self.checkAuxBorrow(self.lowerNibble(self.registers['A']),self.lowerNibble(self.registers[reg]))
        self.registers['A'] = res
        self.checkBorrow('A')
        self.checkSign()
        if self.registers['F']['CY']:
            res = hex(res).replace('0x','')
            self.registers['A'] = int(res[1:],16)
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()
         
    def Sui(self,val):
        res = self.registers['A'] + (~val+1)
        self.checkAuxBorrow(self.lowerNibble(self.registers['A']),self.lowerNibble(val))
        self.registers['A'] = res
        self.checkBorrow('A')
        self.checkSign()
        if self.registers['F']['CY']:
            res = hex(res).replace('0x','')
            self.registers['A'] = int(res[1:],16)
        self.checkParity()
        self.checkZero()

    def Ana(self,reg):
        res = self.registers['A'] & self.registers[reg]
        self.registers['A'] = res
        res = bin(res).replace('0b','').zfill(8)
        self.setSign(res[0]=='1')
        self.registers['F']['CY'] = 0
        self.registers['F']['AC'] = 1
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()

    def Ani(self,val):
        res = self.registers['A'] & val
        self.registers['A'] = res
        res = bin(res).replace('0b','').zfill(8)
        self.setSign(res[0]=='1')
        self.registers['F']['CY'] = 0
        self.registers['F']['AC'] = 1
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()

    def Ora(self,reg):
        res = self.registers['A'] | self.registers[reg]
        self.registers['A'] = res
        res = bin(res).replace('0b','').zfill(8)
        self.setSign(res[0]=='1')
        self.registers['F']['CY'] = 0
        self.registers['F']['AC'] = 0
        self.checkParity()
        self.checkZero()

    def Ori(self,val):
        res = self.registers['A'] | val
        self.registers['A'] = res
        res = bin(res).replace('0b','').zfill(8)
        self.setSign(res[0]=='1')
        self.registers['F']['CY'] = 0
        self.registers['F']['AC'] = 1
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()

    def Xra(self,reg):
        res = self.registers['A'] ^ self.registers[reg]
        self.registers['A'] = res
        res = bin(res).replace('0b','').zfill(8)
        self.setSign(res[0]=='1')
        self.registers['F']['CY'] = 0
        self.registers['F']['AC'] = 1
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()

    def Xri(self,val):
        res = self.registers['A'] ^ val
        self.registers['A'] = res
        res = bin(res).replace('0b','').zfill(8)
        self.setSign(res[0]=='1')
        self.registers['F']['CY'] = 0
        self.registers['F']['AC'] = 1
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()

    def Cma(self):
        val = bin(self.registers['A']).replace('0b','').zfill(8)
        res = ''
        for i in val:
            if i=='0':
                res += '1'
            else:
                res += '0'
        self.registers['A']=int(res,2)

    def Cmc(self):
        if self.registers['F']['CY']:
            self.registers['F']['CY'] = 0
        else:
            self.registers['F']['CY'] = 1
        self.setFlagRegister()

    def Cmp(self,reg):
        val = self.getRegister(reg)
        res = self.Sui(val)
        if res < 0:
            self.registers['F']['CY'] = 1
            self.registers['F']['Z'] = 0
        elif res == val:
            self.registers['F']['CY'] = 0
            self.registers['F']['Z'] = 1
        else:
            self.registers['F']['CY'] = 0
            self.registers['F']['Z'] = 0
        self.setFlagRegister()

    def Cpi(self,val):
        res = self.Sui(val)
        if res < 0:
            self.registers['F']['CY'] = 1
            self.registers['F']['Z'] = 0
        elif res == val:
            self.registers['F']['CY'] = 0
            self.registers['F']['Z'] = 1
        else:
            self.registers['F']['CY'] = 0
            self.registers['F']['Z'] = 0
        self.setFlagRegister()

    def Dad(self,regpair):
        byte = self.getRegisterPair(regpair)
        res = self.getRegisterPair('H') + byte
        if res > 0xffff:
            self.registers['F']['CY'] = 1
            res = hex(res).replace('0x','')
            self.setRegisterPair('H',int(res[1:],16))
        else:
            self.registers['F']['CY'] = 0
        self.setFlagRegister()

    def Inr(self,reg):
        self.checkAuxCarry(self.lowerNibble(self.registers[reg]),1)
        if self.registers[reg] < 0xff:
            self.registers[reg] += 1
        elif self.registers[reg] == 0xff:
            self.registers[reg] = 0x00
        self.checkCarry(reg)
        if self.registers['F']['CY']:
            res = hex(self.registers[reg]).replace('0x','')
            self.registers[reg] = int(res[1:],16)
            self.registers['F']['CY']=0
        self.checkAll()
        self.setFlagRegister()

    def Dcr(self,reg):
        self.checkAuxBorrow(self.lowerNibble(self.registers[reg]),1)
        if self.registers[reg] > 0x00:
            self.registers[reg] -= 1
        elif self.registers[reg] == 0x00:
            self.registers[reg] = 0xff
        self.checkBorrow(reg)
        self.checkSign()
        if self.registers['F']['CY']:
            res = hex(self.registers[reg]).replace('0x','')
            self.registers[reg] = int(res[1:],16)
            self.registers['F']['CY']=0
        self.checkParity()
        self.checkZero()
        self.setFlagRegister()

    def Inx(self,reg):
        reg2 = self.regPair[reg]
        if self.registers[reg2] == 0xff:
            self.registers[reg2] = 0
            if self.registers[reg] == 0xff:
                self.registers[reg] = 0
            else:
                self.registers[reg] += 1
        else:
            self.registers[reg2] += 1

    def Dcx(self,reg):
        reg2 = self.regPair[reg]
        if self.registers[reg2] == 0x00:
            self.registers[reg2] = 0xff
            if self.registers[reg] == 0x00:
                self.registers[reg] = 0xff
            else:
                self.registers[reg] -= 1
        else:
            self.registers[reg2] -= 1

    def Ral(self):
        val = bin(self.registers['A']).replace('0b','').zfill(8)
        val = str(self.registers['F']['CY'])+val
        val = bin(int(val,2) << 1).replace('0b','')
        self.registers['F']['CY'] = int(val[0],2)
        self.registers['A'] = int(val[1:],2)
        self.setFlagRegister()

    def Rar(self):
        val = bin(self.registers['A']).replace('0b','').zfill(8)
        val = val + str(self.registers['F']['CY'])
        val = bin(int(val,2) >> 1).replace('0b','').zfill(9)
        self.registers['F']['CY'] = int(val[8],2)
        self.registers['A'] = int(val[:8],2)
        self.setFlagRegister()

    def Rlc(self):
        val = self.registers['A']
        bit = val & 0x80
        val <<= 1
        if bit != 0:
            val |= 1
            self.setCarry()
        val &= 0xFF
        self.registers['A'] = val
        self.setFlagRegister()

    def Rrc(self):
        val = self.registers['A']
        bit = val & 0x01
        val >>= 1
        if bit != 0:
            val |= 0x80
            self.setCarry()
        val &= 0xFF
        self.registers['A'] = val
        self.setFlagRegister()



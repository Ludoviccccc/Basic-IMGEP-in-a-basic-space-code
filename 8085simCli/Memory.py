# DATA LIMITED TO ffff(65535)
# DATA STORED AS DECIMAL
# RANGE OF MEMORY FROM BASE ADDRESS TO (BASE ADDRESS + LENGTH OF DATA LIST)
# DATA ACCEPTED AS INTEGER OR HEX


class RAM():

    def __init__(self,baseAddr,sizeInK):

        self.baseAddr = baseAddr
        self.data = [0]*sizeInK*1024

    def Write(self,addr,data):
        try:
            data = hex(data).replace('0x','')
            if len(data) <= 2:
                self.data[addr-self.baseAddr] = int(data,16)
            else:
                if len(data) == 3:
                    data = str(0) + data[:]
                if len(data) > 4:
                    return
                self.data[addr-self.baseAddr] = int(data[:2],16)
                self.data[addr-self.baseAddr+1] = int(data[2:],16)
        except TypeError:
            return
                     

    def Show(self):
        addr = self.baseAddr
        for i in self.data:
            if i!=0:
                print(hex(addr).replace('0x','')+": "+str(i))
            addr+=1

    def Read(self,addr):
        return self.data[addr-self.baseAddr]

    def ShowRange(self,start,end):
        addr = start
        while addr <= end:
            print(hex(addr).replace('0x','').zfill(4)+": "+str(self.Read(addr)))
            addr+=1

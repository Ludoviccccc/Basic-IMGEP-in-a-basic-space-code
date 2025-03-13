

class Bus:

    def __init__(self):

        self.mem_peripheral = {}
        self.io_peripheral = {}
        self.io_port = []

    def setMemoryPeripheral(self,peripheral,start,end):
        self.mem_peripheral[peripheral] = (start,end)
        

    def setIOPeripheral(self,peripheral,port_addr):
        self.io_peripheral[peripheral] = self.io_port
        if port_addr < 0xff:
            self.io_peripheral[peripheral].append(port_addr)
        

    def ReadMemory(self,addr):
        for k,v in self.mem_peripheral.items():
            if addr >= v[0] and addr <= v[1]:
                return k.Read(addr)

    def WriteMemory(self,addr,data):
        for k,v in self.mem_peripheral.items():
            if addr >= v[0] and addr <= v[1] and data <= 0xFF:
                k.Write(addr,data)

    def ReadIO(self,addr):
        for k,v in self.io_peripheral.items():
            if addr in v:
                return k.Read(addr)
                

    def WriteIO(self,addr,data):
        for k,v in self.io_peripheral.items():
            if addr in v and data <= 0xff:
                k.Write(addr,data)
        

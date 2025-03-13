
class IO:

    def __init__(self,a,b,c):

        self.a = [a,0]
        self.b = [b,0]
        self.c = [c,0]

    def Read(self,addr):
        if addr == self.a[0]:
            return self.a[1]
        elif addr == self.b[0]:
            return self.b[1]
        elif addr == self.c[0]:
            return self.c[1]
        else:
            return

    def Write(self,addr,data):
        if addr == self.a[0]:
            self.a[1] = data
        elif addr == self.b[0]:
            self.b[1] = data
        elif addr == self.c[0]:
            self.c[1] = data
        else:
            return

    def ClearOutput(self):
        self.a[1] = 0
        self.b[1] = 0
        self.c[1] = 0

    def Show(self):
        print("Output port A: "+ hex(self.a[1]))
        print("Output port B: "+ hex(self.b[1]))
        print("Output port C: "+ hex(self.c[1]))

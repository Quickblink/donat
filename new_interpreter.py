from itertools import count

class InvalidOpCodeException(Exception):
    pass


class Cpu:
    def __init__(self, ram_size, pd, code):
        self.ram_pointer = 0
        self.pd_pointer = 0
        self.program_counter = 0
        self.ram = [0] * ram_size
        self.pd = pd
        self.code = code
        self.pd_is_head = True
        self.register = 0
        self.logical = False

    def readValue(self):
        return self.pd[self.pd_pointer] if self.pd_is_head else self.ram[self.ram_pointer]

    def movePointer(self, instruction):
        amount = -1 if instruction == 'LEFT' else 1
        if self.pd_is_head:
            self.pd_pointer = (self.pd_pointer + amount) % len(self.pd)
        else:
            self.ram_pointer = (self.ram_pointer + amount) % len(self.ram)

    # empty: 2, 3, 5, 12, 15
    def clk(self):
        instruction = self.code[self.program_counter]
        if instruction in ['LEFT', 'RIGHT']:  # LEFT, RIGHT
            self.movePointer(instruction)
        elif instruction == 'SWH':  # SWH
            self.pd_is_head = not self.pd_is_head
        elif instruction == 'LOAD':  # LOAD
            self.register = self.readValue()
            self.logical = (self.register != 0)
        elif instruction == 'ADD':  # ADD
            self.register += self.readValue()
            self.logical = (self.register < 0)
        elif instruction == 'INV':  # INV
            self.register = -self.register
            self.logical = not self.logical
        elif instruction == 'WRITE':  # WRITE
            if self.pd_is_head:
                self.pd[self.pd_pointer] = self.register
            else:
                self.ram[self.ram_pointer] = self.register
        elif instruction == 'CJMP':  # CJMP
            if self.logical:
                self.program_counter = self.pd[self.pd_pointer]
                return
        elif instruction == 'MJMP':  # MJMP
            self.pd_pointer = self.readValue()
            self.logical = True
        elif instruction == 'SHL':  # SHL
            self.register = self.register * 2
            self.logical = 0 #should catch leftover in praxis
        elif instruction == 'SHR':  # SHR
            self.register = self.register // 2
            self.logical = self.register % 2
        else:
            raise InvalidOpCodeException
        self.program_counter += 1

    def print_state(self, end='\n', i=None):
        print((f'{i: >6}' if i is not None else '') +
              f'{self.program_counter: >6} '
              f'{self.ram_pointer: >3} '
              f'{self.pd_pointer: >3}  '
              f'{self.register: >6} '
              f'{" pd" if self.pd_is_head else "ram"} '
              f'{self.ram} '
              f'{self.pd}',
              end=end)

    def run(self):
        self.print_state(i=0)
        for i in count(1):
            self.clk()
            self.print_state(i=i)
            if self.program_counter >= len(self.code):
                break


if __name__ == '__main__':
    code = [
        1, 4, 6, 14, 8, 1, 10, 6, 1, 7, 9, 0, 6, 13, 0, 6, 4, 0, 10
    ]
    cpu = Cpu(3, [0, 12], code)
    cpu.ram[0] = 2
    cpu.ram[1] = 10000

    cpu.print_state()
    while cpu.program_counter < len(cpu.code):
        cpu.clk()
        cpu.print_state('')
        input()


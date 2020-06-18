from itertools import count

class InvalidOpCodeException(Exception):
    pass

'''
TODO:
-faster movement
-direct jumps
-jmp on both memories
-handle input and output
-array mode
-use copy for existing array implementation
-build more comfortable assembler
'''


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

        self.out_buffer = 0
        self.new_output = False
        self.in_buffer = 0
        self.new_input = False


    def readValue(self):
        return self.pd[self.pd_pointer] if self.pd_is_head else self.ram[self.ram_pointer]

    def writeValue(self, value):
        if self.pd_is_head:
            self.pd[self.pd_pointer] = value
        else:
            self.ram[self.ram_pointer] = value


    # empty: 2, 3, 5, 12, 15
    def clk(self):
        instruction = self.code[self.program_counter]
        move_dict = {'LEFT': -1, 'RIGHT': 1, 'LEFT4': -4, 'RIGHT4': 4}
        if instruction in move_dict:  # LEFT, RIGHT, LEFT4, RIGHT4
            if self.pd_is_head:
                self.pd_pointer = (self.pd_pointer + move_dict[instruction])  # % len(self.pd)
            else:
                self.ram_pointer = (self.ram_pointer + move_dict[instruction])  # % len(self.ram)
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
            self.writeValue(self.register)
        elif instruction in ['JMP', 'CJMP']:  # CJMP
            if instruction == 'JMP' or self.logical:
                self.program_counter = self.readValue() #self.pd[self.pd_pointer]
                return
        elif instruction == 'MJMP':  # MJMP
            self.pd_pointer = self.readValue()
            #self.logical = True
            self.pd_is_head = True
        elif instruction == 'SHIFT':  # SHL
            if self.pd_is_head:
                self.register = self.register * 2
                self.logical = 0 #should catch leftover in praxis
            else:
                self.logical = self.register % 2
                self.register = self.register // 2
        elif instruction == 'COPY':
            if self.pd_is_head:
                self.ram[self.ram_pointer] = self.readValue()
            else:
                self.pd[self.pd_pointer] = self.readValue()
            self.ram_pointer += 1
            self.pd_pointer += 1
        elif instruction == 'INP':
            if not self.new_input:
                return
            self.writeValue(self.in_buffer)
            self.new_input = False
        elif instruction == 'OUT':
            if self.new_output:
                return
            self.out_buffer = self.readValue()
            self.new_output = True
        else:
            raise InvalidOpCodeException
        self.program_counter += 1

    def print_state(self, end='\n', i=None):
        vars = {}
        for name, loc in self.alloc_var.items():
            vars[name] = self.ram[loc+self.ram_offset]
        print((f'{i: >6}' if i is not None else '') +
              f'{self.program_counter: >6} '
              f'{self.code[self.program_counter]: >6} '
              f'{self.ram_pointer: >3} '
              f'{self.pd_pointer: >3}  '
              f'{self.register: >6} '
              f'{"T"if self.logical else "F"} '
              f'{" pd" if self.pd_is_head else "ram"} '
              f'{str(vars)}'
              f'{self.ram} '
              f'{self.pd}',
              end=end)

    def run(self):
        self.print_state(i=0)
        for i in count(1):
            self.clk()
            if self.program_counter >= len(self.code):
                break
            self.print_state(i=i)


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


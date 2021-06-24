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
'''


class Cpu:
    def __init__(self, ram_size, pd, code):
        self.dyn_pointer = 0
        self.stat_pointer = 0
        self.program_counter = 0
        self.dyn_mem = [0] * ram_size
        self.stat_mem = pd
        self.code = code
        self.register = 0
        self.logical = False
        self.bus = 0

        self.hold = False

        self.in_buffer = 0

    # NON, LOAD, ADD, SUB, SHL, SHR, ADDL, ADDO, ZERO
    # INT, JMP, JN, JNN, JZ, JNZ, JL, JNL
    # = 17
    #ZERO Bus state?

    def exControl(self, ins):
        # CNT, INT, JMP, JN, JNN, JZ, JNZ, JNS
        if self.hold:
            return
        elif ins == 'CNT':
            pass
        elif ins == 'INT':
            self.hold = True
        elif ins in ['JMP', 'JN', 'JNN', 'JZ', 'JNZ', 'JNS']:
            do_jump = True
            if ins == 'JN':
                do_jump = self.register < 0
            elif ins == 'JNN':
                do_jump = not (self.register < 0)
            elif ins == 'JZ':
                do_jump = self.register == 0
            elif ins == 'JNZ':
                do_jump = self.register != 0
            elif ins == 'JNS':
                do_jump = not self.logical
            if do_jump:
                self.program_counter = self.bus
                return
        else:
            raise InvalidOpCodeException('control code '+ins)
        self.program_counter += 1

    def exRegister(self, ins):
        # NON, LOAD, ADD, SUB, SHL, SHR
        if ins == 'LOAD':
            self.register = self.bus
        elif ins == 'ADD':
            self.register += self.bus
        elif ins == 'SUB':
            self.register -= self.bus
        elif ins == 'SHL':
            self.register *= 2
            #TODO: add leftover
        elif ins == 'SHR':
            self.logical = self.register % 2
            self.register = self.register // 2
        elif ins != 'NON':
            raise InvalidOpCodeException('register code '+ins)

    def exMoveDynamic(self, ins):
        move_dict = {'LEFT': -1, 'RIGHT': 1, 'LEFT4': -4, 'RIGHT4': 4}
        if ins in move_dict:
            self.dyn_pointer += move_dict[ins]
        elif ins != 'HLD':
            raise InvalidOpCodeException('move dyn code '+ins)

    def exMoveStatic(self, ins):
        move_dict = {'LEFT': -1, 'RIGHT': 1, 'LEFT4': -4, 'RIGHT4': 4}
        if ins in move_dict:
            self.stat_pointer += move_dict[ins]
        elif ins == 'MJMP':
            self.stat_pointer = self.bus
        elif ins != 'HLD':
            raise InvalidOpCodeException('move dyn code '+ins)

    def exBusTransaction(self, ins):
        #D: Dynamic Memory
        #S: Static Memory
        #R: Register
        #I: Input
        # D2R, S2R, D2S, S2D, R2D, R2S, I2R, I2D
        if ins in ['D2R', 'D2S']:
            self.bus = self.dyn_mem[self.dyn_pointer]
        elif ins in ['S2R', 'S2D']:
            self.bus = self.stat_mem[self.stat_pointer]
        elif ins in ['R2D', 'R2S']:
            self.bus = self.register #TODO: check order of ops
        elif ins in ['I2R', 'I2D']:
            self.bus = self.in_buffer
        else:
            raise InvalidOpCodeException('bus code '+ins)

        if ins in ['S2D', 'R2D', 'I2D']:
            self.dyn_mem[self.dyn_pointer] = self.bus #TODO: movement before this? answer: this belongs together, but movement before that
        elif ins in ['D2S', 'R2S']:
            self.stat_mem[self.stat_pointer] = self.bus

    def clk(self):
        ins_ctl, ins_reg, ins_mvd, ins_mvs, ins_bus = self.code[self.program_counter]
        self.exControl(ins_ctl)
        self.exRegister(ins_reg)
        self.exMoveDynamic(ins_mvd)
        self.exMoveStatic(ins_mvs)
        self.exBusTransaction(ins_bus)
        

    def print_state(self, end='\n', i=None):
        vars = {}
        for name, loc in self.alloc_var.items():
            vars[name] = self.dyn_mem[loc + self.ram_offset]
        print((f'{i: >6}' if i is not None else '') +
              f'{self.program_counter: >6} '
              f'{self.code[self.program_counter]: >6} '
              f'{self.dyn_pointer: >3} '
              f'{self.stat_pointer: >3}  '
              f'{self.register: >6} '
              f'{"T"if self.logical else "F"} '
              f'{" pd" if self.pd_is_head else "ram"} '
              f'{str(vars)}'
              f'{self.dyn_mem} '
              f'{self.stat_mem}',
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
    cpu.dyn_mem[0] = 2
    cpu.dyn_mem[1] = 10000

    cpu.print_state()
    while cpu.program_counter < len(cpu.code):
        cpu.clk()
        cpu.print_state('')
        input()


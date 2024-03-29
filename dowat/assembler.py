

class Assembler:
    def __init__(self, functions):
        self.const = [10000, 'main_adress'] #10k = exit address for main
        self.code = ['SWH', 'RIGHT', 'SWH', 'RIGHT', 'MJMP', 'LOAD', 'CJMP']
        self.code = [
            ('CNT', 'NON', 'RIGHT', 'RIGHT', 'S2R'),
            ('CNT', 'NON', 'HLD', 'MJMP', 'S2R'),
            ('JMP', 'NON', 'HLD', 'RIGHT', 'D2R'), #TODO: RIGHT?

        ]
        self.function_signatures = {}
        self.functions = functions

    def assemble(self):
        self.assemble_function('main')
        self.const[1] = self.function_signatures['main']
        return self.const, self.code

    #TODO: simple recursion
    def assemble_function(self, name):
        if name not in self.functions:
            raise Exception('function missing: '+name)
        for const in self.functions[name]['const']: # make sure all subcalls are assembled
            if type(const) is tuple and const[0] == 'fun' and const[1] not in self.function_signatures and const[1] != name:
                self.assemble_function(const[1])
        self.function_signatures[name] = len(self.const) + self.functions[name]['entry']
        assembled_code = []
        label_dict = {}
        for cmd in self.functions[name]['code']:
            if type(cmd) is str:
                assembled_code.append(cmd)
            else:
                label_dict[cmd[0]] = len(assembled_code)
        assembled_const = []
        for const in self.functions[name]['const']:
            if type(const) is tuple:
                if const[0] == 'fun':
                    assembled_const.append(self.function_signatures[const[1]])
                if const[0] == 'label':
                    if type(const[1]) is int:
                        assembled_const.append(const[1] + len(self.code))
                    else:
                        assembled_const.append(label_dict[const[1]] + len(self.code))
                if const[0] == 'mem_jmp':
                    assembled_const.append(const[1] + len(self.const))
            else:
                assembled_const.append(const)
        self.const += assembled_const
        self.code += assembled_code
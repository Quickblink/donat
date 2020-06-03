


some_functions = {
    'MUL': {
        'code': [
            'LOAD',
            'ADD'
        ],
        'const': [
            ('fun', 'DIV'),  # mem_position of function
            ('label', 5),  # relative code position to start
            ('mem_jmp', 2),  # relative mem position to start
            5
        ],
        'entry': 0 # pos in mem with start label
    }
}


class Assembler:
    def __init__(self, functions):
        self.const = ['main_adress']
        self.code = ['MJMP', 'CJMP']
        self.function_signatures = {}
        self.functions = functions

    def assemble(self):
        self.assemble_function('main')
        self.const[0] = self.function_signatures['main']

    #TODO: simple recursion
    def assemble_function(self, name):
        if name not in self.functions:
            raise Exception('function missing: '+name)
        for const in self.functions[name]['const']: # make sure all subcalls are assembled
            if type(const) is tuple and const[0] == 'fun' and const[1] not in self.function_signatures:
                self.assemble_function(const[1])
        assembled_const = []
        for const in self.functions[name]['const']:
            if type(const) is tuple:
                if const[0] == 'fun':
                    assembled_const.append(self.function_signatures[const[1]])
                if const[0] == 'label':
                    assembled_const.append(const[1] + len(self.code))
                if const[0] == 'mem_jmp':
                    assembled_const.append(const[1] + len(self.const))
            else:
                assembled_const.append(const)
        self.function_signatures[name] = len(self.const) + self.functions[name]['entry']
        self.const += assembled_const
        self.code += self.functions[name]['code']
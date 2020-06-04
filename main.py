from compiler import from_source
from assembled_programs import Assembler, some_functions
from new_interpreter import Cpu
import random

random.seed(0)
array = list(range(100))
random.shuffle(array)

functions = some_functions
hl_functions = from_source('test_program2.py')
for k, v in hl_functions.items():
    functions[k] = v
const, code = Assembler(functions).assemble()

for i, cmd in enumerate(functions['main']['code']):
    print(i, cmd)

ar_pos = len(const)
for i in range(len(const)):
    if const[i] == 123456:
        const[i] = len(const)
const += array

cpu = Cpu(100, const, code)
cpu.alloc_var = functions['main']['var_alloc']
cpu.ram_offset = 1
for f in functions:
    print(f, functions[f]['const'])
print(functions['main']['const'])
cpu.run()

#print(functions)
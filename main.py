from compiler import from_source
from assembled_programs import Assembler, some_functions
from new_interpreter import Cpu
import random

random.seed(0)
array = list(range(10))
random.shuffle(array)

functions = some_functions
hl_functions = from_source('test_program2.py')
for k, v in hl_functions.items():
    functions[k] = v
const, code = Assembler(functions).assemble()

count_dict = {}
for i, cmd in enumerate(functions['teile_babe']['code']):
    #print(i, cmd)
    if cmd not in count_dict:
        count_dict[cmd] = 1
    else:
        count_dict[cmd] += 1

for cmd in count_dict:
    count_dict[cmd] /= len(functions['teile_babe']['code'])


print(count_dict)

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
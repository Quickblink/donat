import ast
from new_interpreter import Cpu


class Babeception(Exception):
    pass



def from_source(program_source):
    code = open(program_source).read()  # a = b*b'
    func_list = ast.parse(code).body
    func_dict = {}
    for f in func_list:
        if type(f) is not ast.FunctionDef:
            raise Babeception('Only function defs allowed in source!')
        func_dict[f.name] = compile_function(f)
    return func_dict

def compile_function(node):
    parameters = []
    for arg in node.args.args:
        parameters.append(arg.arg)
    fC = functionCompiler()
    code = fC.compile_statements(node.body)
    code = fC.manage_array_jump(code)
    alloc_ram, const_alloc, const_size = simple_allocate(code, parameters)
    mp = MovementPlanner(alloc_ram, const_alloc)
    assembly, translate_label = mp.plan(code)
    const_mem = [0] * const_size
    const_mem[0] = ('label', 0)
    for k, v in const_alloc['const'].items():
        const_mem[v] = k
    for k, v in const_alloc['label'].items():
        const_mem[v] = ('label', translate_label[k])
    for k, v in const_alloc['fun'].items():
        const_mem[v] = ('fun', k)
    #for line in code:
   #     print(line)
   # for line in assembly:
    #    print(line)
    #print(const_mem)
    #print(alloc_ram)
    # print(const_alloc, const_size)
    # print(translate_label)
    return {'code': assembly, 'const': const_mem, 'entry': 0}


'''
stmt = FunctionDef(identifier name, arguments args,
                   stmt* body, expr* decorator_list, expr? returns,
                   string? type_comment)

arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
             expr* kw_defaults, arg? kwarg, expr* defaults)

arg = (identifier arg, expr? annotation, string? type_comment)
'''


class MovementPlanner:
    def __init__(self, alloc_ram, alloc_const):
        self.alloc_ram = alloc_ram
        self.alloc_const = alloc_const
        self.ins_ptr = 0
        self.ram_ptr = 0
        self.const_ptr = 0
        self.const_head = True
        self.out_code = []
        self.translate_label = {}
        self.label_ram = {}
        self.label_const_head = {}
        self.unreachable = False


    def add_instructions(self, instructions):
        self.out_code += instructions
        self.ins_ptr += len(instructions)

    def lookup(self, target):
        lookup = self.alloc_ram if target[1] in ['tmp', 'var'] else self.alloc_const
        return lookup[target[1]][target[0]], target[1] not in ['tmp', 'var']

    def move_to_target(self, target, lookup=False):
        #target = self.lookup(target) if lookup else target
        if target[1] == 'array':
            self.move_to_target(target[0])
            self.add_instructions(['MJMP'])
            self.const_head = True
        if len(target) == 3:
            self.move_to_target(target[2])
            self.add_instructions(['MJMP'])
            self.const_head = True
        if target[1] in ['const', 'label', 'fun', 'mem_jmp']:
            if not self.const_head:
                self.add_instructions(['SWH'])
                self.const_head = True
            goal_pos = self.alloc_const[target[1]][target[0]]
            direction = 'LEFT' if goal_pos <= self.const_ptr else 'RIGHT'
            distance = abs(goal_pos - self.const_ptr)
            self.add_instructions([direction]*distance)
        if target[1] in ['tmp', 'var']:
            if self.const_head:
                self.add_instructions(['SWH'])
                self.const_head = False
            goal_pos = self.alloc_ram[target[1]][target[0]]
            direction = 'LEFT' if goal_pos <= self.ram_ptr else 'RIGHT'
            distance = abs(goal_pos - self.ram_ptr)
            self.add_instructions([direction]*distance)


    def move_to_state(self, state, set_logicbit):
        if self.unreachable:
            self.unreachable = False
            return
        goal_ram, goal_const, goal_head = state
        if len(goal_const) == 3:
            self.move_to_target(goal_const)
        if goal_head: # end on const
            if self.alloc_ram[goal_ram[1]][goal_ram[0]] != self.ram_ptr:
                self.move_to_target(goal_ram)
            self.move_to_target(goal_const)
            if set_logicbit:
                self.add_instructions(['LOAD'])
        else: # end on ram
            if self.alloc_const[goal_const[1]][goal_const[0]] != self.const_ptr:
                self.move_to_target(goal_const)
                if set_logicbit:
                    self.add_instructions(['LOAD'])
            self.move_to_target(goal_ram)


    def decide_label_state(self, code):
        for i, cmd in enumerate(code):
            if cmd[0] == 'label':
                for inn_cmd in code[i:]:
                    if len(inn_cmd) == 2 and type(inn_cmd[1]) is tuple:
                        if inn_cmd[1][1] == 'tmp' or inn_cmd[1][1] == 'var':
                            self.label_ram[cmd[1]] = inn_cmd[1]
                            break
                        elif len(inn_cmd[1]) == 3:
                            self.label_ram[cmd[1]] = inn_cmd[1][2]
                        else:
                            self.label_const_head[cmd[1]] = True

    def plan(self, code):
        self.decide_label_state(code)
        for cmd in code:
            if cmd[0] in ['LOAD', 'ADD', 'WRITE', 'INV'] and len(cmd) == 2:
                if len(cmd) == 2:
                    self.move_to_target(cmd[1])
                self.add_instructions([cmd[0]])
            elif cmd[0] in ['CJMP', 'dir_jump', 'label']:
                if cmd[1] not in self.label_ram:
                    self.label_ram[cmd[1]] = self.ram_ptr
                self.move_to_state((self.label_ram[cmd[1]], cmd[1], self.label_const_head[cmd[1]]), cmd[0] == 'dir_jump')
                if cmd[0] == 'label':
                    self.translate_label[cmd[1][0]] = self.ins_ptr
                else:
                    self.add_instructions('CJMP')
            elif cmd[0] == 'label_back':
                self.translate_label[cmd[1][0]] = self.ins_ptr
                self.const_ptr = self.alloc_const[cmd[1][1]][cmd[1][0]]
                self.ram_ptr -= 1
                self.const_head = True
            elif cmd[0] == 'call':
                self.move_to_state(((cmd[2][1], 'tmp'), (cmd[1], 'fun'), True), False)
                self.add_instructions(['MJMP', 'CJMP'])
            elif cmd[0] == 'return':
                if self.const_head:
                    self.add_instructions(['SWH'])
                    self.const_head = False
                self.add_instructions(['LEFT'] * (self.ram_ptr+1))
                self.add_instructions(['MJMP', 'CJMP'])
                self.unreachable = True
            else:
                raise Babeception('Assembler Error: ' + str(cmd) + ' unkown.')
        return self.out_code, self.translate_label




# make variables static again, because of nonlinear control flow
def simple_allocate(code, parameters):
    tmp_current = 0#len(parameters)
    tmp_max = tmp_current
    const_alloc = {'const': {}, 'label': {}, 'fun': {}, 'mem_jmp': {}}
    ram_alloc = {'var': {}, 'tmp': {}}
    const_cnt = 1
    for i, v in enumerate(parameters):
        ram_alloc['var'][v] = i
    var_cnt = len(parameters)
    for cmd in code:
        if len(cmd) == 2 and type(cmd[1]) is tuple:
            if cmd[1][1] == 'tmp':
                if cmd[0] == 'WRITE':
                    ram_alloc[cmd[1][0]] = tmp_current
                    tmp_current += 1
                    tmp_max = max(tmp_max, tmp_current)
                else:
                    tmp_current -= 1
            if cmd[1][1] in ['const', 'mem_jmp']:
                if cmd[1][0] not in const_alloc[cmd[1][1]]:
                    const_alloc[cmd[1][1]][cmd[1][0]] = const_cnt
                    const_cnt += 1
            if cmd[1][1] == 'var':
                if cmd[1][0] not in ram_alloc['var']:
                    ram_alloc['var'][cmd[1][0]] = var_cnt
                    var_cnt += 1
            if len(cmd) == 3:
                tmp_current -= 1
                    #tmp_current += 1
                    #tmp_max = max(tmp_max, tmp_current)
        if cmd[0] == 'call':
            #if cmd[1] not in function_signatures:
            #    compile_function(cmd[1])
            tmp_current -= len(cmd[2])
            #tmp_max = max(tmp_max, tmp_current+function_signatures[cmd[1]][0])
            const_alloc['fun'][cmd[1]] = const_cnt
            const_cnt += 1
        if cmd[0] in ['label', 'label_back']:
            if cmd[1][0] not in const_alloc['label']:
                const_alloc['label'][cmd[1][0]] = const_cnt
                const_cnt += 1
    for tmp in ram_alloc['tmp']:
        ram_alloc['tmp'][tmp] += var_cnt
    return ram_alloc, const_alloc, const_cnt


class functionCompiler:
    def __init__(self):
        self.tmp_cnt = 0
        self.jmp_cnt = 0

    def manage_array_jump(self, code):
        # have to save: 'const' or 'label' gets used, 'label' itself
        # no save: another 'array', 'call', 'return'
        full_code = []
        snippet = []
        active = False
        for cmd in code:
            # add load/write adress before access (if nötig) and add remove tmp before corresponding next const action
            if len(cmd) == 2 and type(cmd[1]) is tuple and cmd[1][1] in ['const', 'label'] and active:
                ncode = [
                    ('LOAD', (cmd[1], 'mem_jmp')),
                    ('WRITE', (self.tmp_cnt, 'tmp'))
                ]
                new_target = (*cmd[1], (self.tmp_cnt, 'tmp'))
                full_code += ncode + snippet + [(cmd[0], new_target)]
                self.tmp_cnt += 1
                active = False
            else:
                if len(cmd) == 2 and type(cmd[1]) is tuple and cmd[1][1] == 'array':
                    if active:
                        full_code += snippet
                        snippet = []
                    active = True
                elif cmd[0] in ['call', 'return'] and active:
                    full_code += snippet
                    snippet = []
                    active = False
                if active:
                    snippet += [cmd]
                else:
                    full_code += [cmd]
        return full_code + snippet



    '''
    stmt =  Return(expr? value)
          | Assign(expr* targets, expr value, string? type_comment)
          | Expr(expr value)

          | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
    '''
    #TODO: add augassign statements (for arrays)
    #TODO: merge different cases of if and loops
    def compile_statements(self, node_list):
        full_code = []
        for node in node_list:
            if type(node) is ast.Assign:
                if type(node.targets[0]) is ast.Name:
                    target = (node.targets[0].id, 'var')
                elif type(node.targets[0]) is ast.Subscript:
                    code1, var1 = self.compile_int_expr(ast.BinOp(node.targets[0].value, ast.Add(), node.targets[0].slice.value), None)
                    full_code += code1
                    target = (var1, 'array')
                else:
                    raise Babeception(str(node.targets[0])+' not supported as target of assignment')
                code = self.compile_int_expr(node.value, target)
                full_code += code
            elif type(node) is ast.Return:
                code = self.compile_int_expr(node.value, 'register')
                op = ('return',)
                full_code += code + [op]
            elif type(node) is ast.Expr:
                pass  # TODO: useful only with arrays, empty target?
            elif type(node) is ast.If:
                if len(node.orelse) == 0:
                    test, _ = preprocess_boolean(node.test, True)
                    end = (self.jmp_cnt, 'label')
                    self.jmp_cnt += 1
                    logic_code = self.compile_boolean(test, end)
                    body_code = self.compile_statements(node.body)
                    full_code += logic_code + body_code + [('label', end)]
                else:
                    test, _ = preprocess_boolean(node.test, False)
                    end = (self.jmp_cnt, 'label')
                    body = (self.jmp_cnt + 1, 'label')
                    self.jmp_cnt += 2
                    logic_code = self.compile_boolean(test, body)
                    else_code = self.compile_statements(node.orelse)
                    body_code = self.compile_statements(node.body)
                    full_code += logic_code + else_code + [('dir_jump', end), ('label', body)] + body_code + [('label', end)]
            elif type(node) is ast.While:
                test_pre, _ = preprocess_boolean(node.test, False)
                test = (self.jmp_cnt, 'label')
                body = (self.jmp_cnt + 1, 'label')
                self.jmp_cnt += 2
                body_code = self.compile_statements(node.body)
                logic_code = self.compile_boolean(test_pre, body)
                full_code += [('dir_jump', test), ('label', body)] + body_code + [('label', test)] + logic_code
            elif type(node) is ast.For:
                # target, iter, body
                if type(node.target) is not ast.Name or type(node.iter) is not ast.Tuple:
                    raise Babeception('Wrong For Syntax!')
                code = self.compile_int_expr(node.iter.elts[0], 'register')
                full_code += code
                test = (self.jmp_cnt, 'label')
                body = (self.jmp_cnt + 1, 'label')
                self.jmp_cnt += 2
                test_pre = ast.Compare(node.target, [ast.Lt()], [node.iter.elts[1]])
                body_code = self.compile_statements(node.body)
                logic_code = self.compile_boolean(test_pre, body)
                inc_code = self.compile_int_expr(node.iter.elts[2], 'register')
                inc_code2 = [
                    ('ADD', (node.target.id, 'var')),
                    ('WRITE', (node.target.id, 'var'))
                ]
                full_code += [('WRITE', (node.target.id, 'var')), ('dir_jump', test), ('label', body)]\
                             + body_code + inc_code + inc_code2 + [('label', test)] + logic_code

        return full_code

    '''
    expr = BinOp(expr left, operator op, expr right)
         | UnaryOp(unaryop op, expr operand)
         | Call(expr func, expr* args, keyword* keywords)
         | Constant(constant value, string? kind)
         | Subscript(expr value, slice slice, expr_context ctx)
         | Name(identifier id, expr_context ctx)
    
    slice = Index(expr value)
    operator = Add | Sub | Mult | Div | Mod | Pow | LShift
                 | RShift | MatMult
    unaryop = USub
    
    '''
    def save_output(self, prev_code, target):
        if target == 'register':
            return prev_code
        elif target:
            return prev_code + [('WRITE', target)]
        else:
            new_target = (self.tmp_cnt, 'tmp')
            self.tmp_cnt += 1
            return prev_code + [('WRITE', new_target)], new_target

    def compile_int_expr(self, node, target):
        if type(node) is ast.BinOp:
            op_to_func = {ast.MatMult: 'MUL', ast.Div: 'DIV', ast.Mod: 'MOD', ast.Pow: 'POW'}
            if type(node.op) in op_to_func:
                return self.compile_int_expr(ast.Call(ast.Name(op_to_func[type(node.op)]), [node.left, node.right]), target)
            # TODO: add shift, maybe second mul?
            elif type(node.op) is ast.Add:
                code_right, var1 = self.compile_int_expr(node.right, False)
                code_left = self.compile_int_expr(node.left, 'register')
                op = ('ADD', var1)
                return self.save_output(code_right+code_left+[op], target)
            elif type(node.op) is ast.Sub:
                return self.compile_int_expr(ast.BinOp(ast.UnaryOp(ast.USub(), node.right), ast.Add(), node.left), target)
            else:
                raise Babeception(str(type(node.op))+' is not supported! '+str(node.op))
        elif type(node) is ast.UnaryOp:
            if type(node.op) is ast.USub:
                code_prev = self.compile_int_expr(node.operand, 'register') #TODO: check for negative constants
                op = ('INV',)
                return self.save_output(code_prev+[op], target)
            else:
                raise Babeception(str(type(node.op)) + ' is not supported!')
        elif type(node) is ast.Call:
            jmp_code = self.jmp_cnt
            code = [
                ('LOAD', (jmp_code, 'label')),
                ('WRITE', (self.tmp_cnt, 'tmp'))
            ]
            self.jmp_cnt += 1
            old_cnt = self.tmp_cnt+1
            self.tmp_cnt += len(node.args)+1
            for i in range(len(node.args)):
                part_code = self.compile_int_expr(node.args[i], (old_cnt+i, 'tmp'))
                code += part_code
            code += [
                ('call', node.func.id, list(range(old_cnt-1, old_cnt + len(node.args)))),  # maybe start location is enough
                ('label_back', (jmp_code, 'label'))
            ]
            return self.save_output(code, target)
        elif type(node) in [ast.Constant, ast.Name, ast.Subscript]:
            code = []
            if type(node) is ast.Constant:
                par = (node.value, 'const')
            elif type(node) is ast.Name:
                par = (node.id, 'var')
            else:
                code1, var1 = self.compile_int_expr(ast.BinOp(node.value, ast.Add(), node.slice.value), None)
                code += code1
                par = (var1, 'array')
            load = ('LOAD', par)
            if target == 'register':
                return code+[load]
            elif target:
                write = ('WRITE', target)
                return code+[load, write]
            else:
                return code, par
        else:
            raise Babeception(str(type(node)) + ' is not supported!')


    def compile_boolean(self, expr, jump_target):
        if type(expr) is ast.BoolOp:
            if type(expr.op) is ast.And:
                code = []
                end = (self.jmp_cnt, 'label')
                self.jmp_cnt += 1
                for v in expr.values:
                    part_code = self.compile_boolean(v, end)
                    code += part_code
                part_code = self.compile_boolean(expr.pos_child, jump_target)
                code += part_code
                code += [('label', end)]
                return code
            elif type(expr.op) is ast.Or:
                code = []
                for v in expr.values:
                    part_code = self.compile_boolean(v, jump_target)
                    code += part_code
                return code
            else:
                raise Babeception(str(type(expr.op)) + ' is not supported in boolean!')
        elif type(expr) is ast.UnaryOp:
            if type(expr.op) is ast.Not:
                part_code = self.compile_boolean(expr.operand, None)
                return part_code+[('INV',)]+do_jump(jump_target)
            else:
                raise Babeception(str(type(expr.op)) + ' is not supported in boolean!')
        elif type(expr) is ast.Compare: #TODO: make triple comparisons
            if type(expr.ops[0]) in [ast.NotEq, ast.Lt, ast.Gt]:
                left = expr.comparators[0] if type(expr.ops[0]) is ast.Gt else expr.left
                right = expr.left if type(expr.ops[0]) is ast.Gt else expr.comparators[0]
                code_a, par = self.compile_int_expr(left, None)
                code_b = self.compile_int_expr(right, 'register')
                code = [
                    ('INV',),
                    ('ADD', par)
                ]
                if type(expr.ops[0]) is ast.NotEq:
                    code += [
                        ('WRITE', (self.tmp_cnt, 'tmp')),
                        ('LOAD', (self.tmp_cnt, 'tmp'))
                    ]
                    self.tmp_cnt += 1
                return code_a+code_b+code+do_jump(jump_target)
            else:
                raise Babeception('Error should be in preprocessing')
        elif type(expr) is ast.Call:
            code_a = self.compile_int_expr(expr, 'register')
            return code_a+do_jump(jump_target)
        elif type(expr) is ast.Name:
            return [('LOAD', (expr.id, 'var'))]+do_jump(jump_target)
        else:
            raise Babeception('Error should be in preprocessing')


'''
expr = BoolOp(boolop op, expr* values)
     | UnaryOp(unaryop op, expr operand)
     | Compare(expr left, cmpop* ops, expr* comparators)
     | Call(expr func, expr* args, keyword* keywords)
     | Name(identifier id, expr_context ctx)

boolop = And | Or
unaryop = Not
cmpop = Eq | NotEq | Lt | LtE | Gt | GtE
'''
def preprocess_child(expr, inv):
    return (ast.UnaryOp(ast.Not(), expr), True) if inv else (expr, False)

def preprocess_boolean(expr, inv):
    if type(expr) is ast.BoolOp:
        if type(expr.op) is ast.And or type(expr.op) is ast.Or:
            new_vs = []
            pos_child = None
            for v in expr.values:
                nv, is_inv = preprocess_boolean(v, inv)
                new_vs.append(nv)
                pos_child = pos_child if is_inv else nv
            pos_child = pos_child or new_vs[0]
            if (type(expr.op) is ast.And and inv) or (type(expr.op) is ast.Or and not inv):
                return ast.BoolOp(ast.Or(), new_vs), False
            new_vs.remove(pos_child)
            new_vs2 = []
            for v in new_vs:
                nv, _ = preprocess_boolean(v, True)
                new_vs2.append(nv)
            expr = ast.BoolOp(ast.And(), new_vs)
            expr.pos_child = pos_child
            return expr, False
        else:
            raise Babeception(str(type(expr.op)) + ' is not supported in boolean!')
    elif type(expr) is ast.UnaryOp:
        if type(expr.op) is ast.Not:
            return preprocess_boolean(expr.operand, False) if inv else preprocess_boolean(expr.operand, True)
        else:
            raise Babeception(str(type(expr.op)) + ' is not supported in boolean!')
    elif type(expr) is ast.Compare:
        if type(expr.ops[0]) is ast.NotEq or type(expr.ops[0]) is ast.Lt or type(expr.ops[0]) is ast.Gt:
            return preprocess_child(expr, inv)
        elif type(expr.ops[0]) is ast.Eq:
            return preprocess_boolean(ast.UnaryOp(ast.Not(), ast.Compare(expr.left, [ast.NotEq()], expr.comparators)), inv)
        elif type(expr.ops[0]) is ast.LtE:
            return preprocess_boolean(ast.UnaryOp(ast.Not(), ast.Compare(expr.left, [ast.Gt()], expr.comparators)), inv)
        elif type(expr.ops[0]) is ast.GtE:
            return preprocess_boolean(ast.UnaryOp(ast.Not(), ast.Compare(expr.left, [ast.Lt()], expr.comparators)), inv)
        else:
            raise Babeception(str(type(expr.ops[0])) + ' is not supported in boolean!')
    elif type(expr) is ast.Call or type(expr) is ast.Name:
        return preprocess_child(expr, inv)
    else:
        raise Babeception(str(type(expr)) + ' is not supported in boolean!')


def do_jump(jump_target):
    return [('CJMP', jump_target)] if jump_target else []


'''
if __name__ == '__main__':
    clicke_babe = False
    code = open('test_program2.py').read()  # a = b*b'
    tree = ast.parse(code).body[0]
    cpu = Cpu(*something(tree))
    #code, _, _ = compile_statements(tree, 0, 0)
    #for line in code:
    #    print(line)
    #print(count_temps(code))
    #cpu = Cpu(3, [0, 12], code)
    cpu.ram[0] = 500*28
    cpu.ram[1] = 500*21
    if clicke_babe:
        cpu.print_state()
        while cpu.program_counter < len(cpu.code):
            cpu.clk()
            cpu.print_state('')
            input()
    else:
        cpu.run()

'''
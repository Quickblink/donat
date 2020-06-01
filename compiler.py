import ast
from new_interpreter import Cpu

function_signatures = {'MUL': (5,15)} #size, const_address


class Babeception(Exception):
    pass

'''
stmt = FunctionDef(identifier name, arguments args,
                   stmt* body, expr* decorator_list, expr? returns,
                   string? type_comment)

arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
             expr* kw_defaults, arg? kwarg, expr* defaults)

arg = (identifier arg, expr? annotation, string? type_comment)
'''


def compile_function(name):
    pass

def something(node):
    parameters = []
    for arg in node.args.args:
        parameters.append(arg.arg)
    code, _, _ = compile_statements(node.body, 0, 0)
    size, vars, const_alloc, consts, funcs, const_size = count_temps(code, parameters)
    alloc_ram = simple_allocate_ram(size, vars, code)
    assembly, translate_label = simple_assembler(alloc_ram, const_alloc, code)
    const_mem = [0] * const_size
    # TODO: set functions own address to const_mem[0]
    for k, v in const_alloc['const'].items():
        const_mem[v] = k
    for k, v in const_alloc['label'].items():
        const_mem[v] = translate_label[k]
    for k, v in const_alloc['fun'].items():
        const_mem[v] = function_signatures[k][1]
    for line in code:
        print(line)
    for line in assembly:
        print(line)
    print(const_mem)
    print(alloc_ram)
    #print(const_alloc, const_size)
    #print(translate_label)
    return size+len(vars), const_mem, assembly


def move(ram_ptr, const_ptr, const_head, target, forceswitch=False):
    code = []
    current_pos = ram_ptr if target[1] else const_ptr
    if current_pos == target[0] and not forceswitch:
        return code, ram_ptr, const_ptr, const_head
    if target[1] == const_head:  # True = RAM :(
        code.append('SWH')
        const_head = not const_head
    direction = 'LEFT' if target[0] <= current_pos else 'RIGHT'
    distance = abs(target[0] - current_pos)
    code += [direction] * distance
    ram_ptr, const_ptr = (target[0], const_ptr) if target[1] else (ram_ptr, target[0])
    return code, ram_ptr, const_ptr, const_head

def lookup_target(alloc_ram, alloc_const, target):
    lookup = alloc_ram if target[1] in ['tmp', 'var'] else alloc_const
    return lookup[target[1]][target[0]], target[1] in ['tmp', 'var']  # True = RAM :(


def simple_assembler(alloc_ram, alloc_const, code):
    ins_ptr = 0
    ram_ptr = 0
    const_ptr = 0
    const_head = True
    out_code = []
    translate_label = {}
    label_ram = {}
    label_const_head = {}
    for i, cmd in enumerate(code):
        if cmd[0] == 'label':
            for inn_cmd in code[i:]:
                if len(inn_cmd) == 2 and type(inn_cmd[1]) is tuple:
                    if inn_cmd[1][1] == 'tmp' or inn_cmd[1][1] == 'var':
                        label_ram[cmd[1]] = alloc_ram[inn_cmd[1][1]][inn_cmd[1][0]]
                        break
                    else:
                        label_const_head[cmd[1]] = True
    for i, cmd in enumerate(code):
        if cmd[0] in ['INV']:
            out_code.append(cmd[0])
            ins_ptr += 1
        elif cmd[0] in ['LOAD', 'ADD', 'WRITE'] and len(cmd) == 2:
            ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, lookup_target(alloc_ram, alloc_const, cmd[1]), forceswitch=True)
            out_code += ins_code + [cmd[0]]
            ins_ptr += len(ins_code) + 1
        elif cmd[0] in ['CJMP', 'dir_jump']:
            if cmd[1][1] not in label_ram:
                label_ram[cmd[1][1]] = ram_ptr
            if cmd[1][1] in label_const_head:
                ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, (label_ram[cmd[1][1]], True))
                out_code += ins_code
                ins_ptr += len(ins_code)
            ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, lookup_target(alloc_ram, alloc_const, tuple(reversed(cmd[1]))), forceswitch=(cmd[0] == 'dir_jump' or cmd[1][1] in label_const_head))
            out_code += ins_code
            ins_ptr += len(ins_code)
            if cmd[0] == 'dir_jump':
                out_code.append('LOAD')
                ins_ptr += 1
            if cmd[1][1] not in label_const_head:
                ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, (label_ram[cmd[1][1]], True), forceswitch=True)
                out_code += ins_code
                ins_ptr += len(ins_code)
            out_code.append('CJMP')
            ins_ptr += 1
        elif cmd[0] == 'label':
            if cmd[1] not in label_ram:
                label_ram[cmd[1]] = ram_ptr
            if cmd[1] in label_const_head:
                ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, (label_ram[cmd[1]], True))
                out_code += ins_code
                ins_ptr += len(ins_code)
            ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, lookup_target(alloc_ram, alloc_const, tuple(reversed(cmd))), forceswitch=(cmd[1] in label_const_head))
            out_code += ins_code
            ins_ptr += len(ins_code)
            if cmd[1] not in label_const_head:
                ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, (label_ram[cmd[1]], True), forceswitch=True)
                out_code += ins_code
                ins_ptr += len(ins_code)
            translate_label[cmd[1]] = ins_ptr
        elif cmd[0] == 'call':
            ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, lookup_target(alloc_ram, alloc_const, (cmd[2][1], 'tmp')))
            out_code += ins_code
            ins_ptr += len(ins_code)
            ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, lookup_target(alloc_ram, alloc_const, (cmd[1], 'fun')), forceswitch=True)
            out_code += ins_code
            ins_ptr += len(ins_code)
            out_code += [
                'MJMP',
                'CJMP'
            ]
            ins_ptr += 2
        elif cmd[0] == 'return':
            ins_code, ram_ptr, const_ptr, const_head = move(ram_ptr, const_ptr, const_head, (-1, True), forceswitch=True)
            out_code += ins_code
            ins_ptr += len(ins_code)
            out_code += [
                'MJMP',
                'CJMP'
            ]
            ins_ptr += 2
        else:
            raise Babeception('Assembler Error: '+str(cmd)+' unkown.')
    return out_code, translate_label


def simple_allocate_ram(size, vars, code):
    allocation = {'var': {}, 'tmp': []}
    for i, v in enumerate(vars):
        allocation['var'][v] = i
    slots = list(range(len(vars)+size-1, len(vars)-1, -1))
    for cmd in code:
        if len(cmd) == 2 and type(cmd[1]) is tuple:
            if cmd[1][1] == 'tmp':
                if cmd[0] == 'WRITE':
                    s = slots.pop()
                    allocation['tmp'].append(s)
                    assert(len(allocation['tmp']) == cmd[1][0]+1)
                else:
                    slots.append(allocation['tmp'][cmd[1][0]])
                    assert(len(slots) == 1 or slots[-1] + 1 == slots[-2])
        if cmd[0] == 'call':
            assert(len(slots) >= function_signatures[cmd[1]][0]-len(cmd[2]))
            for p in reversed(cmd[2]):
                slots.append(allocation['tmp'][p])
                assert (slots[-1] + 1 == slots[-2] or len(slots) == 1)
    return allocation

# make variables static again, because of nonlinear control flow
def count_temps(code, parameters):
    tmp_current = 0#len(parameters)
    tmp_max = tmp_current
    consts = []
    funcs = []
    const_alloc = {'const': {}, 'label': {}, 'fun': {}}
    const_cnt = 1
    vars = []+parameters
    for cmd in code:
        if len(cmd) == 2 and type(cmd[1]) is tuple:
            if cmd[1][1] == 'tmp':
                if cmd[0] == 'WRITE':
                    tmp_current += 1
                    tmp_max = max(tmp_max, tmp_current)
                else:
                    tmp_current -= 1
            if cmd[1][1] == 'const':
                if cmd[1][0] not in consts:
                    consts.append(cmd[1][0])
                    const_alloc['const'][cmd[1][0]] = const_cnt
                    const_cnt += 1
            if cmd[1][1] == 'var':
                if cmd[1][0] not in vars:
                    vars.append(cmd[1][0])
                    #tmp_current += 1
                    #tmp_max = max(tmp_max, tmp_current)
        if cmd[0] == 'call':
            if cmd[1] not in funcs:
                funcs.append(cmd[1])
            if cmd[1] not in function_signatures:
                compile_function(cmd[1])
            tmp_current -= len(cmd[2])
            tmp_max = max(tmp_max, tmp_current+function_signatures[cmd[1]][0])
            const_alloc['fun'][cmd[1]] = const_cnt
            const_cnt += 1
        if cmd[0] == 'label':
            if cmd[1] not in const_alloc['label']:
                const_alloc['label'][cmd[1]] = const_cnt
                const_cnt += 1
    return tmp_max, vars, const_alloc, consts, funcs, const_cnt

'''
expr = BinOp(expr left, operator op, expr right)
     | UnaryOp(unaryop op, expr operand)
     | Call(expr func, expr* args, keyword* keywords)
     | Constant(constant value, string? kind)
     | Subscript(expr value, slice slice, expr_context ctx)
     | Name(identifier id, expr_context ctx)

slice = Index(expr value)
operator = Add | Sub | Mult | Div | Mod | Pow
unaryop = USub

'''
# TODO: make parameter type different from tmp, maybe dont
def save_output(code_prev, cnt, target):
    if target == 'register':
        return code_prev, cnt
    elif target:
        write = ('WRITE', target)
        return code_prev + [write], cnt
    else:
        write = ('WRITE', (cnt, 'tmp'))
        return code_prev + [write], cnt + 1, (cnt, 'tmp')

def compile_int_expr(node, cnt, target):
    if type(node) is ast.BinOp:
        if type(node.op) is ast.Mult:
            return compile_int_expr(ast.Call(ast.Name('MUL'), [node.left, node.right]), cnt, target)
        # TODO: add div, mod, pow
        elif type(node.op) is ast.Add:
            code_right, cnt, var1 = compile_int_expr(node.right, cnt, False)
            code_left, cnt = compile_int_expr(node.left, cnt, 'register')
            op = ('ADD', var1)
            return save_output(code_right+code_left+[op], cnt, target)
        elif type(node.op) is ast.Sub:
            return compile_int_expr(ast.BinOp(ast.UnaryOp(ast.USub(), node.right), ast.Add(), node.left), cnt, target)
        else:
            raise Babeception(str(type(node.op))+' is not supported! '+str(node.op))
    elif type(node) is ast.UnaryOp:
        if type(node.op) is ast.USub:
            code_prev, cnt = compile_int_expr(node.operand, cnt, 'register') #TODO: check for negative constants
            op = ('INV',)
            return save_output(code_prev+[op], cnt, target)
        else:
            raise Babeception(str(type(node.op)) + ' is not supported!')
    elif type(node) is ast.Call:
        code = [
            ('LOAD', ('label', None)),
            ('WRITE', (cnt, 'tmp'))
        ]
        old_cnt = cnt+1
        cnt += len(node.args)+1
        for i in range(len(node.args)):
            part_code, cnt = compile_int_expr(node.args[i], cnt, (old_cnt+i, 'tmp'))
            code += part_code
        code += [
            #('move', (old_cnt, 'tmp')),
            ('call', node.func.id, list(range(old_cnt-1, old_cnt + len(node.args)))),  # maybe start location is enough
            #('MJMP', ('func_entry', node.func.id)),
            #('CJMP',),
            ('label', None) # TODO: get a jump_code
        ]
        return save_output(code, cnt, target)
    elif type(node) is ast.Constant or type(node) is ast.Name:
        par = (node.value, 'const') if type(node) is ast.Constant else (node.id, 'var')
        load = ('LOAD', par)
        if target == 'register':
            return [load], cnt
        elif target:
            write = ('WRITE', target)
            return [load, write], cnt
        else:
            return [], cnt, par
    elif type(node) is ast.Subscript:
        pass #TODO: implement arrays
    else:
        raise Babeception(str(type(node)) + ' is not supported!')

'''
stmt =  Return(expr? value)
      | Assign(expr* targets, expr value, string? type_comment)
      | Expr(expr value)

      | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
      | While(expr test, stmt* body, stmt* orelse)
      | If(expr test, stmt* body, stmt* orelse)
'''
def compile_statements(node_list, int_cnt, jmp_cnt):
    full_code = []
    for node in node_list:
        if type(node) is ast.Assign:
            if type(node.targets[0]) is ast.Name:
                target = (node.targets[0].id, 'var')
            elif type(node.targets[0]) is ast.Subscript:
                target = None #TODO: implement arrays
            code, int_cnt = compile_int_expr(node.value, int_cnt, target)
            full_code += code
        elif type(node) is ast.Return:
            code, int_cnt = compile_int_expr(node.value, int_cnt, 'register')
            op = ('return',)
            full_code += code# + [op]
        elif type(node) is ast.Expr:
            pass #TODO: useful only with arrays, empty target?
        elif type(node) is ast.If:
            if len(node.orelse) == 0:
                test, _ = preprocess_boolean(node.test, True)
                end = ('label', jmp_cnt)
                jmp_cnt += 1
                logic_code, jmp_cnt, int_cnt = compile_boolean(test, end, jmp_cnt, int_cnt)
                body_code, int_cnt, jmp_cnt = compile_statements(node.body, int_cnt, jmp_cnt)
                full_code += logic_code+body_code+[end]
            else:
                test, _ = preprocess_boolean(node.test, False)
                end = ('label', jmp_cnt)
                body = ('label', jmp_cnt+1)
                jmp_cnt += 2
                logic_code, jmp_cnt, int_cnt = compile_boolean(test, body, jmp_cnt, int_cnt)
                else_code, int_cnt, jmp_cnt = compile_statements(node.orelse, int_cnt, jmp_cnt)
                body_code, int_cnt, jmp_cnt = compile_statements(node.body, int_cnt, jmp_cnt)
                full_code += logic_code+else_code+[('dir_jump', end), body]+body_code+[end]
        elif type(node) is ast.While:
            test_pre, _ = preprocess_boolean(node.test, False)
            test = ('label', jmp_cnt)
            body = ('label', jmp_cnt + 1)
            jmp_cnt += 2
            body_code, int_cnt, jmp_cnt = compile_statements(node.body, int_cnt, jmp_cnt)
            logic_code, jmp_cnt, int_cnt = compile_boolean(test_pre, body, jmp_cnt, int_cnt)
            full_code += [('dir_jump', test), body]+body_code+[test]+logic_code
        elif type(node) is ast.For:
            # target, iter, body
            if type(node.target) is not ast.Name or type(node.iter) is not ast.Tuple:
                raise Babeception('Wrong For Syntax!')
            code, int_cnt = compile_int_expr(node.iter.elts[0], int_cnt, 'register')
            full_code += code
            test = ('label', jmp_cnt)
            body = ('label', jmp_cnt + 1)
            jmp_cnt += 2
            test_pre = ast.Compare(node.target, [ast.Lt()], [node.iter.elts[1]])
            body_code, int_cnt, jmp_cnt = compile_statements(node.body, int_cnt, jmp_cnt)
            logic_code, jmp_cnt, int_cnt = compile_boolean(test_pre, body, jmp_cnt, int_cnt)
            inc_code, int_cnt = compile_int_expr(node.iter.elts[2], int_cnt, 'register')
            inc_code2 = [
                ('ADD', (node.target.id, 'var')),
                ('WRITE', (node.target.id, 'var'))
            ]
            full_code += [('WRITE', (node.target.id, 'var')), ('dir_jump', test), body] + body_code+inc_code+inc_code2+[test]+logic_code



    return full_code, int_cnt, jmp_cnt


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

def compile_boolean(expr, jump_target, cnt, tmp_cnt):
    if type(expr) is ast.BoolOp:
        if type(expr.op) is ast.And:
            code = []
            end = ('label', cnt)
            cnt += 1
            for v in expr.values:
                part_code, cnt, tmp_cnt = compile_boolean(v, end, cnt, tmp_cnt)
                code += part_code
            part_code, cnt, tmp_cnt = compile_boolean(expr.pos_child, jump_target, cnt, tmp_cnt)
            code += part_code
            code += [end]
            return code, cnt, tmp_cnt
        elif type(expr.op) is ast.Or:
            code = []
            for v in expr.values:
                part_code, cnt, tmp_cnt = compile_boolean(v, jump_target, cnt, tmp_cnt)
                code += part_code
            return code, cnt, tmp_cnt
        else:
            raise Babeception(str(type(expr.op)) + ' is not supported in boolean!')
    elif type(expr) is ast.UnaryOp:
        if type(expr.op) is ast.Not:
            part_code, cnt, tmp_cnt = compile_boolean(expr.operand, None, cnt, tmp_cnt)
            return part_code+[('INV',)]+do_jump(jump_target), cnt, tmp_cnt
        else:
            raise Babeception(str(type(expr.op)) + ' is not supported in boolean!')
    elif type(expr) is ast.Compare: #TODO: make triple comparisons
        if type(expr.ops[0]) is ast.NotEq:
            code_a, tmp_cnt, par = compile_int_expr(expr.left, tmp_cnt, None)
            code_b, tmp_cnt = compile_int_expr(expr.comparators[0], tmp_cnt, 'register')
            code = [
                ('INV',),
                ('ADD', par),
                ('WRITE', (tmp_cnt, 'tmp')),
                ('LOAD', (tmp_cnt, 'tmp'))
            ]
            return code_a+code_b+code+do_jump(jump_target), cnt, tmp_cnt+1
        elif type(expr.ops[0]) is ast.Lt:
            code_a, tmp_cnt, par = compile_int_expr(expr.left, tmp_cnt, None)
            code_b, tmp_cnt = compile_int_expr(expr.comparators[0], tmp_cnt, 'register')
            code = [
                ('INV',),
                ('ADD', par)
            ]
            return code_a+code_b+code+do_jump(jump_target), cnt, tmp_cnt
        elif type(expr.ops[0]) is ast.Gt:
            code_a, tmp_cnt, par = compile_int_expr(expr.comparators[0], tmp_cnt, None)
            code_b, tmp_cnt = compile_int_expr(expr.left, tmp_cnt, 'register')
            code = [
                ('INV',),
                ('ADD', par)
            ]
            return code_a+code_b+code+do_jump(jump_target), cnt, tmp_cnt
        else:
            raise Babeception('Error should be in preprocessing')
    elif type(expr) is ast.Call:
        code_a, tmp_cnt = compile_int_expr(expr, tmp_cnt, 'register')
        return code_a+do_jump(jump_target), cnt, tmp_cnt
    elif type(expr) is ast.Name:
        return [('LOAD', (expr.id, 'var'))]+do_jump(jump_target), cnt, tmp_cnt
    else:
        raise Babeception('Error should be in preprocessing')

'''
def compile_function(code):
    tree = ast.parse(code).body[0]
    if type(tree) is ast.Assign:
        return compile_int_expr(tree.value, 0, (tree.targets[0].id, 'var'))
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


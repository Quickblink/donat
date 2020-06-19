

some_functions = {
    'dummy_sample': {
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
    },
    'MUL': {
        'code': [
            'SWH',
            'RIGHT',
            'RIGHT',
            'SWH',
            'RIGHT',
            'COPY',
            'SWH',
            'RIGHT',
            'LEFT4',
            'LOAD',
            'SWH',

            ['start'],
            'SHIFT', #right
            'SWH'
            'WRITE',
            'INV',
            'RIGHT',
            'RIGHT'
            'CJMP',

            'LEFT',
            'LOAD',
            'RIGHT',
            'RIGHT',
            'ADD',
            'WRITE',
            'LEFT',

            ['endif'],
            'LEFT'
            'LOAD',
            'SHL',
            'WRITE',
            'LEFT',
            'LOAD',
            'SWH',
            'CJMP',

            'SWH',
            'RIGHT4',
            'LEFT',
            'LOAD',
            'LEFT4',
            'MJMP',
            'JMP'

        ],
        'const' : [('label', 0), ('label', 'endif'), ('label', 'start')],
        'entry' : 0

    }
}
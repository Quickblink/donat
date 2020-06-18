

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
            'LOAD',
            'RIGHT',
            'RIGHT',

            'WRITE',
            'SWH',
            'RIGHT',
            'LOAD',
            'SWH',

            'LEFT',
            'WRITE',
            'SWH',
            'RIGHT',
            'LOAD'
            
            'RIGHT',
            'LOAD',
            'SWH',
            'RIGHT4',
            'WRITE',

            'SWH',
            'RIGHT',
            'LOAD',
            'SWH',
            'LEFT',

            'WRITE',
            'LEFT4',
            'RIGHT',
            'LOAD',
            'SHR',

            'WRITE',
            'INV',
            'RIGHT',

            'CJMP',
            'LOAD',
            'RIGHT',
            'ADD',
            'WRITE',

            'LEFT',
            'LOAD',
            'SHL',
            'WRITE',
            'LEFT',

            'LOAD',
            'SWH',
            'LEFT',
            'CJMP',
            'SWH',

            'RIGHT',
            'RIGHT',
            'LOAD',
            'LEFT',
            'LEFT',

            'LEFT',
            'MJMP',
            'SWH',
            'WRITE',
            'SWH',

            'CJMP'

        ],
        'const' : [('label', 0), ('label', 4), ('label', 16)],
        'entry' : 0

    }
}
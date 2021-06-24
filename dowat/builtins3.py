

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
            'upper',
            ('CNT', 'NON', 'RIGHT', 'HLD', 'S2R'),
            ('JNS', 'NON', 'HLD', 'RIGHT', 'D2R'),

            ('CNT', 'LOAD', 'RIGHT', 'HLD', 'D2R'),
            ('CNT', 'ADD', 'HLD', 'HLD', 'R2D'),
            ('CNT', 'NON', 'LEFT', 'HLD', 'D2R'),

            'endif',
            ('CNT', 'LOAD', 'HLD', 'HLD', 'x'),
            ('CNT', 'SHL', 'HLD', 'HLD', 'R2D'),
            ('CNT', 'NON', 'LEFT', 'HLD', 'D2R'),

            'entry', #TODO: entrypoint static position
            ('CNT', 'LOAD', 'HLD', 'HLD', 'S2R'),
            ('JNZ', 'SHR', 'HLD', 'LEFT', 'R2D'),

            ('CNT', 'NON', 'LEFT', 'HLD', 'D2R'),
            ('CNT', 'NON', 'RIGHT4', 'MJMP', 'x'),
            ('CNT', 'NON', 'LEFT', 'HLD', 'D2R'),
            ('CNT', 'LOAD', 'RIGHT', 'HLD', 'S2R'),
            ('JMP', 'NON', 'LEFT4', 'HLD', 'D2R'),

            '''
            ('CNT', 'NON', 'RIGHT', 'HLD', 'x'),
            ('CNT', 'NON', 'RIGHT', 'HLD', 'D2R'),
            ('CNT', 'LOAD', 'LEFT4', 'HLD', 'x'),
            ('CNT', 'NON', 'RIGHT', 'HLD', 'D2R'),
            ('CNT', 'NON', 'HLD', 'MJMP', 'S2R'),
            ('JMP', 'NON', 'HLD', 'HLD', 'D2R'),
            '''

        ],
        'static' : [('label', 'entry'), ('label', 'endif'), ('label', 'upper')],
        'entry' : 0

    }
}
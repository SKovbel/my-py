import numpy

arr = [ # bath
    [ # channels
        [ #row
            [1, 2, 3], #col
            [4, 5, 6] #col
        ],
        [ #row
            [10, 20, 30], #col
            [40, 50, 60] #col
        ],
        [ #row
            [12, 22, 32], #col
            [42, 52, 62] #col
        ]
    ],
    [ # channels
        [ #row
            [11, 12, 13], #col
            [14, 15, 16] #col
        ],
        [ #row
            [11, 12, 13], #col
            [14, 15, 16] #col
        ],
        [ #row
            [11, 12, 13], #col
            [14, 15, 16] #col
        ]
    ]
]
a = numpy.asarray(arr, dtype=numpy.int16)
result = a.transpose(0, 3, 2, 1)
print(result)
exit(0)

arr =[[1,2], [3,4], [5,6]]
a = numpy.asarray(arr, dtype=numpy.int16)
print('matrix:', a)
result = a.transpose()
print('transpose:', result)
result = a.flatten()
print('flatten:', result)
result = numpy.concatenate((a, a))
print('concatenate:', result)


print(numpy.sum(a, axis=0))
print(numpy.sum(a, axis=1))
print(numpy.max(a, axis=1))
print(numpy.min(a, axis=1))
print(numpy.mean(a, axis=1))
print(numpy.var(a, axis=0))
print(numpy.std(a))

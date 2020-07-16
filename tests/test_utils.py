from myautoml.utils import recursive_update


def test_recursive_update():
    a = {'k': 1,
         'l': {'m': 2,
               'n': 3},
         'o': 4}
    b = {'k': 5,
         'l': {'m': 6,
               'p': 7}}

    c = {'k': 5,
         'l': {'m': 6,
               'n': 3,
               'p': 7},
         'o': 4}

    assert recursive_update(a, b) == c

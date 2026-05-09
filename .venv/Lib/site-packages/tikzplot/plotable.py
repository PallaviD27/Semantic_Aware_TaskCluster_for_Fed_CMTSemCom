from abc import ABC
from abc import abstractmethod


class Plotable(ABC):
    '''
    Plotable element for tikzplot
    '''

    def __init__(self, idx, **kwargs):
        self.idx = idx
        self.preamble_lines = []
        self.parameters = {}
        for key in kwargs:
            self.parameters[key] = kwargs[key]

    @abstractmethod
    def generate_tikz(self):
        return None

    def setcolor(self, color):
        # Setup color
        if color is None:
            color_str = 'mycolor{}'.format(self.idx % 6)
        elif isinstance(color, tuple) and len(color) == 3:
            # Setup preamble
            rgb = '{}, {}, {}'.format(*color)
            color_name = 'mycolor{}'.format(str(10**6 + self.idx))
            cstr = r'\definecolor{'+color_name+r'}{rgb}{' + rgb + r'}%'
            self.preamble_lines.append(cstr)
            # Setup color
            color_str = 'mycolor{}'.format(10**6 + self.idx % 6)
        else:
            color_str = color
        self.parameters['color'] = color_str

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

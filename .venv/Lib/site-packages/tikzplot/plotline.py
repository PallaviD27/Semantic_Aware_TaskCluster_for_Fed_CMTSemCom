from .plotable import Plotable


class PlotLine(Plotable):
    '''
    Holds the information for a single plotline
    '''

    def __init__(self, idx, xdata, ydata, linewidth=1.5, color=None,
            label=None, opacity=0.9, style='solid', smooth=True, **kwargs):
        super().__init__(idx, **kwargs)
        self.legend_text = None
        if len(xdata) != len(ydata):
            # Check if all data is numerical
            raise ValueError('xdata and ydata need to have equal length')
        for x, y in zip(xdata, ydata):
            x_numeric = isinstance(x, (float, int))
            y_numeric = isinstance(y, (float, int))
            if (not x_numeric) or (not y_numeric):
                raise ValueError('Both xdata and ydata need to be numerical')
        self.preamble_lines = []
        self.xdata = xdata
        self.ydata = ydata
        self.legend_text = label
        self.style = style
        self.parameters = {}
        self.parameters['smooth'] = smooth
        self.parameters['line width'] = '{}pt'.format(linewidth)
        self.parameters['opacity'] = str(opacity)
        self.setcolor(color)

    def generate_tikz(self):
        '''
        Generate the tikz lines for a given plotline.
        '''
        # Generate the line data
        outlines = []
        options_str = ''
        options = []
        for key in self.parameters:
            value = self.parameters[key]
            if value is True:
                options.append(key)
            else:
                options.append('{}={}'.format(key, value))
        options = list(filter(None, options))
        options_str = ', '.join(options)
        outlines.append(r'\addplot [' + options_str + r']')
        outlines.append(r'table[row sep=crcr]{%')
        xarr, yarr = self.xdata, self.ydata
        for x, y in zip(xarr, yarr):
            outlines.append('{} {}'.format(x, y) + r'\\')
        outlines.append(r'};')
        if self.legend_text is not None:
            legname = self.legend_text
            outlines.append(r'\addlegendentry{' + legname + r'}')
        return outlines

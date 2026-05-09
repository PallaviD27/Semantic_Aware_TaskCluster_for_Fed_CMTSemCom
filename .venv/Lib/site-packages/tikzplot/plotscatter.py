from .plotable import Plotable


class PlotScatter(Plotable):

    def __init__(self, idx, xarr, yarr, label=None, mark='o', color=None, **kwargs):
        super().__init__(idx, **kwargs)
        self.setcolor(color)
        self.parameters['only marks'] = True
        self.parameters['mark'] = mark
        self.xdata = xarr
        self.ydata = yarr
        self.legend_text = label


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


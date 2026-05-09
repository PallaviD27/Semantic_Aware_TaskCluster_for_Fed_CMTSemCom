import json
from .plotable import Plotable
from .plotline import PlotLine
from .plotscatter import PlotScatter
from typing import List
import os
import tempfile



class Plotter():

    def __init__(self):
        '''
        Class to hold drawable object
        '''
        # Includes at the beginning of the document
        self.includes = {}
        #
        parameters = {}
        parameters['width'] = '12.6cm'
        parameters['height'] = '{}cm'.format(round(0.618034 * 12.6, 5))
        parameters['xmin'] = '-10.0'
        parameters['xmax'] = '+10.0'
        parameters['ymin'] = '-10.0'
        parameters['ymax'] = '+10.0'
        parameters['yminorticks'] = 'true'
        # parameters['axis background./style'] = r'{fill=while}'
        parameters['axis x line*'] = 'bottom'
        parameters['axis y line*'] = 'left'
        style = r'{legend cell align=left, align=left, draw=white!15!black}'
        parameters['legend style'] = style
        self.parameters = parameters
        # Get the plot lines
        self.plotables: List[Plotable] = []
        # Internal parameters
        self.xmode = 'linear'
        self.ymode = 'linear'
        self.xlabel = None
        self.ylabel = None

    def update_axis_bounds(self):
        '''
        Automatically updates the axis bounds based on what plots we have.
        '''
        xmin = 1e100
        xmax = -1e100
        ymin = 1e100
        ymax = -1e100
        for plotobj in self.plotables:
            if isinstance(plotobj, (PlotLine, PlotScatter)):
                xarr, yarr = plotobj.xdata, plotobj.ydata
                xmin = min(xmin, min(xarr))
                xmax = max(xmax, max(xarr))
                ymin = min(ymin, min(yarr))
                ymax = max(ymax, max(yarr))
        xwidth = xmax - xmin
        ywidth = xmax - xmin
        xmin = xmin - 0.01*xwidth
        xmax = xmax + 0.01*xwidth
        ymin = ymin - 0.01*ywidth
        ymax = ymax + 0.01*ywidth
        if len(self.plotables) > 0:
            self.parameters['xmax'] = str(xmax)
            self.parameters['xmin'] = str(xmin)
            self.parameters['ymax'] = str(ymax)
            self.parameters['ymin'] = str(ymin)

    def set_xscale(self, mode):
        self.xmode = mode

    def set_yscale(self, mode):
        self.ymode = mode

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def __set__(self, key, value):
        self.parameters[key] = value

    def plot(self, xdata, ydata, **attrs):
        '''
        Add a plotline to the plot
        '''
        new_idx = len(self.plotables)
        line = PlotLine(new_idx, xdata, ydata, **attrs)
        self.plotables.append(line)
        return line

    def scatter(self, xdata, ydata, **attrs):
        '''
        Add a scatter plot
        '''
        new_idx = len(self.plotables)
        obj = PlotScatter(new_idx, xdata, ydata, **attrs)
        self.plotables.append(obj)
        return obj

    def generate_preample(self):
        outlines = []
        outlines.append(r'\documentclass{standalone}')
        outlines.append(r'\usepackage[utf8x]{inputenc}')
        outlines.append(r'\usepackage[T1]{fontenc}')
        outlines.append(r'\usepackage[usenames,dvipsnames]{xcolor} ')
        outlines.append(r'\usepackage[utf8x]{inputenc}')
        outlines.append(r'\usepackage[toc,page]{appendix}')
        outlines.append(r'\usepackage{algorithmicx}')
        outlines.append(r'\usepackage{algorithm}')
        outlines.append(r'\usepackage{algpseudocode}')
        outlines.append(r'\usepackage{amsmath,amssymb,amstext,mathabx}')
        outlines.append(r'\usepackage{amsthm,lipsum}')
        outlines.append(r'\usepackage{amsthm}')
        outlines.append(r'\usepackage{appendix}')
        outlines.append(r'\usepackage{caption} ')
        outlines.append(r'\usepackage{diagbox} ')
        outlines.append(r'\usepackage{enumitem}')
        outlines.append(r'\usepackage{float} ')
        outlines.append(r'\usepackage{graphicx} ')
        outlines.append(r'\usepackage{hyperref}')
        outlines.append(r'\usepackage{lmodern} ')
        outlines.append(r'\usepackage{mathabx}')
        outlines.append(r'\usepackage{mathtools} ')
        outlines.append(r'\usepackage{multicol}')
        outlines.append(r'\usepackage{parcolumns}')
        outlines.append(r'\usepackage{pdfpages} ')
        outlines.append(r'\usepackage{pgfplotstable} ')
        outlines.append(r'\usepackage{pstricks-add}')
        outlines.append(r'\usepackage{relsize} ')
        outlines.append(r'\usepackage{subcaption}')
        outlines.append(r'\usepackage{tcolorbox}')
        outlines.append(r'\usepackage{todonotes}')
        outlines.append(r'\usepackage{verbatim} ')
        outlines.append(r'\usepackage{wasysym}')
        outlines.append(r'\usepackage{tikz}')
        outlines.append(r'\begin{document}')
        outlines.append(r'\definecolor{mycolor0}{rgb}{0.00000,0.44700,0.74100}%')
        outlines.append(r'\definecolor{mycolor1}{rgb}{0.85000,0.32500,0.09800}%')
        outlines.append(r'\definecolor{mycolor2}{rgb}{0.92900,0.69400,0.12500}%')
        outlines.append(r'\definecolor{mycolor3}{rgb}{0.49400,0.18400,0.55600}%')
        outlines.append(r'\definecolor{mycolor4}{rgb}{0.46600,0.67400,0.18800}%')
        outlines.append(r'\definecolor{mycolor5}{rgb}{0.30100,0.74500,0.93300}%')
        # Add generated colors
        for plotobj in self.plotables:
            outlines.extend(plotobj.preamble_lines)
        return outlines

    def generate_tikzstart(self):
        outlines = []
        outlines.append(r'\begin{tikzpicture}')
        outlines.append(r'\begin{axis}[%')
        for key in self.parameters:
            value = self.parameters[key]
            line = '{}={},'.format(key, value)
            outlines.append(line)
        # Set scale and labels
        if self.xmode == 'log':
            outlines.append(r'xmode=log,')
        if self.ymode == 'log':
            outlines.append(r'ymode=log,')
        if self.xlabel is not None:
            outlines.append(r'xlabel={'+ self.xlabel +r'},')
        if self.ylabel is not None:
            outlines.append(r'ylabel={'+ self.ylabel +r'},')
        outlines.append(']')
        return outlines

    def generate_tikzend(self):
        outlines = []
        outlines.append(r'\end{axis}')
        outlines.append(r'\end{tikzpicture}')
        outlines.append(r'\end{document}')
        return outlines

    def save(self, render=True):
        '''
        Export the plot to latex plot
        '''
        self.update_axis_bounds()
        outlines = []
        outlines.extend(self.generate_preample())
        outlines.extend(self.generate_tikzstart())
        for line in self.plotables:
            outlines.extend(line.generate_tikz())
        # Check for appropriate endlines
        outlines.extend(self.generate_tikzend())
        outlines = [line+'\n' for line in outlines]
        # Write lines to output file
        with open('plot.tex', 'w') as fs:
            fs.writelines(outlines)
        # call to compile
        if render:
            os.system('pdflatex plot.tex')
            os.remove('plot.aux')
            os.remove('plot.log')
            os.remove('plot.out')

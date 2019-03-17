from bokeh.io import output_file, show
from bokeh.plotting import figure

"""
    Author: William Blackie

    Class to plot graphs to show training progression in a readable format.
"""


class GraphUtils:
    def __init__(self):
        pass

    @staticmethod
    def create_graph(y, x, legend):
        """
        Method to plot a bokeh graph, used to show training progression.
        :param x: List of X values
        :param y: List of Y values
        :param legend: Name of graph
        :return: None, a html graph will be saved in root project directory.
        """

        # output to static HTML file
        output_file("log_lines.html")

        # create a new plot
        p = figure(
            tools="pan,box_zoom,reset,save",
            y_axis_type="log", y_range=[float(min(y)), float(max(y))], title="Skip-Thought Vectors Training",
            x_axis_label='Iterations (in 100s)', y_axis_label='Cost'
        )

        # render
        p.line(x, y, legend=legend, line_width=3)

        # show the results
        show(p)



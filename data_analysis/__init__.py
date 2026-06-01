"""
data_analysis
=============

A Python toolkit for data cleaning, exploratory data analysis,
feature engineering, and interactive visualisation.

Quick start
-----------
>>> from data_analysis import DataInspector, PlottingMethods

>>> inspector = DataInspector()
>>> inspector.upload_data()          # Google Colab file upload
>>> inspector.handle_missing_values(strategy='median')
>>> inspector.remove_duplicates()
>>> inspector.plot_numerical(['Age', 'Salary'])

>>> PLT = PlottingMethods()
>>> result = PLT.plot_bar_chart(x='Department', y='Salary', data=inspector.df)
>>> PLT.display_image(result)
"""

from .core import DataInspector, PlottingMethods

__all__ = ["DataInspector", "PlottingMethods"]
__version__ = "1.0.0"
__author__ = "E/23/034"

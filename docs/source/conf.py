# Configuration file for the Sphinx documentation builder.

import os
import sys
# Point Sphinx to the root directory where the 'glam_radiomics' package lives.
# This assumes your conf.py is in docs/source/
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'GLAM Radiomics'
copyright = '2026, Yuri S. Velichko, Northwestern University. All Rights Reserved'
author = 'Yuri S. Velichko'
release = '1.0.8'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Pulls documentation from your docstrings
    'sphinx.ext.mathjax',  # Renders the LaTeX equations from your manuscript
    'sphinx.ext.napoleon'  # Supports Google/NumPy style docstrings
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_id': 'G-YKWLCJSXRY',  # Your Google Analytics ID
    'analytics_anonymize_ip': False,
}
html_static_path = ['_static']
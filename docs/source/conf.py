# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'lightning.py'
copyright = '2023-2024, Erik B. Monson'
author = 'Erik B. Monson'

# The full version, including alpha/beta/rc tags
release = 'v2024.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              #'sphinx.ext.graphviz',
              #'sphinx.ext.inheritance_diagram',
              'numpydoc',
              'nbsphinx'
              #'sphinx.ext.napoleon',
]

numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
html_theme_options = {'home_page_in_toc': False}

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
#autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for LaTeX output -------------------------------------------------

# An inelegant solution for getting tqdm's progress bars (in the notebooks) to compile into Latex
latex_elements = {'preamble': r'\DeclareUnicodeCharacter{2588}{\#}'}

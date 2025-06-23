# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Godot Stat Math'
copyright = '2025, edzillion'
author = 'edzillion'
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Additional configuration ------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}

# HTML theme options
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'navigation_depth': 4,
}

# Add custom CSS
html_css_files = [
    'custom.css',
]

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'edzillion',
    'github_repo': 'godot-stat-math',
    'github_version': 'main/docs/',
}

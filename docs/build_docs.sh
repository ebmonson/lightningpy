#!/bin/bash

# Build local local html to check that it looks right
make html

# Build pdf
make latexpdf
cp build/latex/lightningpy.pdf .

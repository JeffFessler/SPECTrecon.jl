#---------------------------------------------------------
# # [SPECTrecon overview](@id 1-overview)
#---------------------------------------------------------

# This page explains the Julia package
# [`SPECTrecon`](https://github.com/JeffFessler/SPECTrecon.jl).

# ### Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using Plots: scatter, plot!, default; default(markerstrokecolor=:auto)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

# ### Overview

# To perform SPECT image reconstruction,
# one must have a model for the imaging system
# encapsulated in a forward project and back projector.

# More details todo


# ### Units

# The pixel dimensions `deltas` can (and should!) be values with units.

# Here is an example ...
using UnitfulRecipes
using Unitful: mm


# ### Methods

# todo

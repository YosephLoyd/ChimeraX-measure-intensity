"""Initialization function to generate ChimeraX Function"""
# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI
from chimerax.core.commands import register
from . import measure_commands

# pylint: disable=abstract-method
# pylint: disable=arguments-differ
# pylint: disable=unused-argument

# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for registering commands,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with BundleInfo and
    # CommandInfo instance instead of command name
    # (when api_version==0)

    # Override method
    @staticmethod
    def register_command(bi, ci, logger):
        """Register command in ChimeraX"""

        if ci.name == "measure distance":
            func = measure_commands.distance_series
            desc = measure_commands.measure_distance_desc
        elif ci.name == "measure intensity":
            func = measure_commands.intensity_series
            desc = measure_commands.measure_intensity_desc
        elif ci.name == "measure composite":
            func = measure_commands.composite_series
            desc = measure_commands.measure_composite_desc
        elif ci.name == "surface recolor":
            func = measure_commands.recolor_surfaces
            desc = measure_commands.recolor_surfaces_desc
        elif ci.name == "surface composite":
            func = measure_commands.recolor_composites
            desc = measure_commands.recolor_composites_desc
        elif ci.name == "measure topology":
            func = measure_commands.topology_series
            desc = measure_commands.measure_topology_desc
        elif ci.name == "measure ridges":
            func = measure_commands.measure_ridges
            desc = measure_commands.measure_ridges_desc
        elif ci.name == "find voids":
            func = measure_commands.voids_seires
            desc = measure_commands.find_voids_desc
        elif ci.name == "void size":
            func = measure_commands.void_size_series
            desc = measure_commands.void_size_desc
        elif ci.name == "void motion":
            func = measure_commands.Void_Motion_series
            desc = measure_commands.Void_Motion_desc
        else:
            raise ValueError(
                f'Trying to register unknown command: {ci.name}')
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, func, logger=logger)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()

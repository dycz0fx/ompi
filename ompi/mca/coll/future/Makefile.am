#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2009-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
coll_future.h \
coll_future_trigger.h \
coll_future_bcast.c \
coll_future_allreduce.c \
coll_future_component.c \
coll_future_module.c \
coll_future_trigger.c \
coll_future_utils.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

component_noinst =
component_install =
if MCA_BUILD_ompi_coll_future_DSO
component_install += mca_coll_future.la
else
component_noinst += libmca_coll_future.la
endif

# See ompi/mca/btl/sm/Makefile.am for an explanation of
# libmca_common_sm.la.

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_future_la_SOURCES = $(sources)
mca_coll_future_la_LDFLAGS = -module -avoid-version
mca_coll_future_la_LIBADD =

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_future_la_SOURCES =$(sources)
libmca_coll_future_la_LDFLAGS = -module -avoid-version

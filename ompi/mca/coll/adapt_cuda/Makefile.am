#
# Copyright (c) 2014      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# $COPYRIGHT$
# 
# Additional copyrights may follow
# 
# $HEADER$
#

dist_ompidata_DATA = help-mpi-coll-adapt-cuda.txt

sources = \
	coll_adapt_cuda_component.c \
	coll_adapt_cuda_module.c \
        coll_adapt_cuda_bcast.c \
        coll_adapt_cuda_reduce.c \
        coll_adapt_cuda.h \
	coll_adapt_cuda_algorithms.h \
	coll_adapt_cuda_context.h \
	coll_adapt_cuda_context.c \
	coll_adapt_cuda_inbuf.c \
	coll_adapt_cuda_inbuf.h \
	coll_adapt_cuda_item.c \
	coll_adapt_cuda_item.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

component_noinst =
component_install =
if MCA_BUILD_ompi_coll_adapt_cuda_DSO
component_install += mca_coll_adapt_cuda.la
else
component_noinst += libmca_coll_adapt_cuda.la
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_adapt_cuda_la_SOURCES = $(sources)
mca_coll_adapt_cuda_la_LDFLAGS = -module -avoid-version
mca_coll_adapt_cuda_la_LIBADD =

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_adapt_cuda_la_SOURCES =$(sources)
libmca_coll_adapt_cuda_la_LDFLAGS = -module -avoid-version

# inc/dg/CMakeLists.txt

# find . -type f \( -name "*.h" -or -name "*.cuh" -or -name "*.hpp" \) -not -path "./exblas/*" -not -path "./matrix/*" -not -path "./geometries/*" -not -path "./file/*" -not -name "*doc.h"| sort
# Manually copy result here and replace ./ with 2 whitespaces
set(DG_HEADERS
  adaptive.h
  advection.h
  algorithm.h
  andersonacc.h
  arakawa.h
  backend/blas1_cuda.cuh
  backend/blas1_dispatch_map.h
  backend/blas1_dispatch_mpi.h
  backend/blas1_dispatch_scalar.h
  backend/blas1_dispatch_shared.h
  backend/blas1_dispatch_vector.h
  backend/blas1_omp.h
  backend/blas1_serial.h
  backend/blas2_cusp.h
  backend/blas2_densematrix.h
  backend/blas2_dispatch_map.h
  backend/blas2_dispatch_mpi.h
  backend/blas2_dispatch_scalar.h
  backend/blas2_dispatch_shared.h
  backend/blas2_dispatch_vector.h
  backend/blas2_selfmade.h
  backend/blas2_sparseblockmat.h
  backend/blas2_stencil.h
  backend/config.h
  backend/densematrix.h
  backend/exblas/accumulate.cuh
  backend/exblas/accumulate.h
  backend/exblas/config.h
  backend/exblas/exblas.h
  backend/exblas/exdot_cuda.cuh
  backend/exblas/exdot_omp.h
  backend/exblas/exdot_serial.h
  backend/exblas/ExSUM.FPE.hpp
  backend/exblas/fpedot_cuda.cuh
  backend/exblas/fpedot_omp.h
  backend/exblas/fpedot_serial.h
  backend/exblas/mpi_accumulate.h
  backend/exblas/mylibm.cuh
  backend/exblas/mylibm.hpp
  backend/exceptions.h
  backend/execution_policy.h
  backend/fma.h
  backend/index.h
  backend/matrix_categories.h
  backend/memory.h
  backend/mpi_cart.h
  backend/mpi_datatype.h
  backend/mpi_gather.h
  backend/mpi_gather_kron.h
  backend/mpi_init.h
  backend/mpi_kron.h
  backend/mpi_matrix.h
  backend/mpi_permutation.h
  backend/mpi_vector.h
  backend/predicate.h
  backend/scalar_categories.h
  backend/sparseblockmat_cpu_kernels.h
  backend/sparseblockmat_gpu_kernels.cuh
  backend/sparseblockmat.h
  backend/sparseblockmat_omp_kernels.h
  backend/sparsematrix_cpu.h
  backend/sparsematrix_gpu.cuh
  backend/sparsematrix.h
  backend/sparsematrix_omp.h
  backend/tensor_traits_cusp.h
  backend/tensor_traits.h
  backend/tensor_traits_scalar.h
  backend/tensor_traits_std.h
  backend/tensor_traits_thrust.h
  backend/timer.h
  backend/typedefs.h
  backend/vector_categories.h
  backend/view.h
  bicgstabl.h
  blas1.h
  blas2.h
  blas.h
  chebyshev.h
  elliptic.h
  enums.h
  eve.h
  extrapolation.h
  functors.h
  helmholtz.h
  implicit.h
  lgmres.h
  multigrid.h
  multistep.h
  multistep_tableau.h
  nullstelle.h
  ode.h
  pcg.h
  pcg_merged.h
  poisson.h
  refined_elliptic.h
  runge_kutta.h
  simpsons.h
  subroutines.h
  tableau.h
  topology/average.h
  topology/base_geometry.h
  topology/base_geometryX.h
  topology/derivativesA.h
  topology/derivatives.h
  topology/derivativesT.h
  topology/derivativesX.h
  topology/dlt.h
  topology/dx.h
  topology/dxX.h
  topology/evaluation.h
  topology/evaluationX.h
  topology/fast_interpolation.h
  topology/fem.h
  topology/fem_weights.h
  topology/filter.h
  topology/functions.h
  topology/geometry.h
  topology/grid.h
  topology/gridX.h
  topology/interpolation.h
  topology/interpolationX.h
  topology/mpi_base_geometry.h
  topology/mpi_derivatives.h
  topology/mpi_evaluation.h
  topology/mpi_grid.h
  topology/mpi_projection.h
  topology/mpi_prolongation.h
  topology/mpi_weights.h
  topology/multiply.h
  topology/operator.h
  topology/operator_tensor.h
  topology/projection.h
  topology/projectionX.h
  topology/prolongation.h
  topology/refined_grid.h
  topology/refined_gridX.h
  topology/split_and_join.h
  topology/stencil.h
  topology/tensor.h
  topology/transform.h
  topology/weights.h
  topology/weightsX.h
  topology/xspacelib.h
)

# These can be compiled for MPI as well
# find . -type f -name "*b.cpp" | sort
set(DG_BENCHMARKS
  arakawa_b.cpp
  blas_b.cpp
  cg2d_b.cpp
  cluster_b.cpp
  elliptic2d_b.cpp
  elliptic3d_b.cpp
  ellipticX2d_b.cpp
  elliptic_b.cpp
  helmholtz_b.cpp
  helmholtzg2_b.cpp
  mpi_matrix_b.cpp
  nvshmem_b.cpp
  nvshmem_device_b.cpp
  multigrid_b.cpp
  poisson_b.cpp
  refined_elliptic2d_b.cpp
)

# These can be compiled for MPI as well
# find . -type f -name "*_t.cpp" | sort
set(DG_TESTS
  adaptive_t.cpp
  arakawa_t.cpp
  backend/fma_t.cpp
  backend/index_t.cpp
  backend/memory_t.cpp
  backend/sparsematrix_t.cpp
  backend/traits_t.cpp
  backend/view_t.cpp
  bicgstabl_t.cpp
  blas1_t.cpp
  blas_t.cpp
  eve_t.cpp
  extrapolation_t.cpp
  helmholtz_t.cpp
  implicit_t.cpp
  multistep_t.cpp
  nullstelle_t.cpp
  pcg_t.cpp
  pcg_merged_t.cpp
  poisson_t.cpp
  runge_kutta_t.cpp
  simpsons_t.cpp
  topology/average_t.cpp
  topology/base_geometry_t.cpp
  topology/derivatives_t.cpp
  topology/derivativesX_t.cpp
  topology/dlt_t.cpp
  topology/dx_t.cpp
  topology/evaluation_t.cpp
  topology/fem_t.cpp
  topology/filter_t.cpp
  topology/geometry_t.cpp
  topology/grid_t.cpp
  topology/interpolation_t.cpp
  topology/multiply_t.cpp
  topology/operator_t.cpp
  topology/operator_tensor_t.cpp
  topology/projection_t.cpp
  topology/refined_grid_t.cpp
  topology/stencil_t.cpp
  topology/tensor_t.cpp
  topology/transform_t.cpp
)

# These are pure MPI tests
# find . -type f -name "*mpit.cpp" | sort
## TODO some mpi tests are missing (see Makefile MPITargets!)
set(DG_MPI_TESTS
  backend/mpi_exblas_mpit.cpp
  backend/mpi_gather_kron_mpit.cpp
  backend/mpi_gather_mpit.cpp
  backend/mpi_init_mpit.cpp
  backend/mpi_kron_mpit.cpp
  backend/mpi_permutation_mpit.cpp
  topology/filter_mpit.cpp
  topology/interpolation_mpit.cpp
  topology/mpi_grid_mpit.cpp
  topology/stencil_mpit.cpp
)

# Manually list tests that can be compiled for MPI
# Cf Makefile, backend/Makefile and topology/Makefile
list( APPEND DG_MPI_TESTS
    topology/evaluation_t.cpp
    topology/dx_t.cpp
    topology/derivatives_t.cpp
    topology/projection_t.cpp
    arakawa_t.cpp
    poisson_t.cpp
    bicgstabl_t.cpp
    pcg_merged_t.cpp
    blas_t.cpp
    blas1_t.cpp
)

# ============================================
# Create DG library target

# Create the dg target library.
# As a header-only library, this must be an interface library.
# This means it has no sources, but can link/be linked to other
# libraries, have dependencies, etc. These properties are then
# inherited by any target that links to it.
add_library(dg_dg INTERFACE)
# When installing the target we prepend the namespace dg::
set_property(TARGET dg_dg PROPERTY EXPORT_NAME dg)
# Create an alias: used to link to the library from external projects.
# consistently with either subdirectory'd and installed library
add_library(feltor::dg::dg ALIAS dg_dg)

# Set the minimum C++ standard
target_compile_features(dg_dg INTERFACE cxx_std_17)

# Create a file set (that can be easily installed)
# This also provides the correct include directory to the target
target_sources( dg_dg INTERFACE
    FILE_SET dg_headers TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/inc
    FILES ${DG_HEADERS}
)


# ============================================
# Add dg dependencies

# Through the exblas library we depend on vcl
# VCL does not use cmake so we have to configure manually
CPMAddPackage(
    NAME vcl
    GITHUB_REPOSITORY vectorclass/version2
    GIT_TAG v2.02.01
    SYSTEM ON
    EXCLUDE_FROM_ALL ON
)

if( vcl_ADDED)
    # with vcl it is recommended to activate and optimize instruction set flags
    message( STATUS "Found Vectorclass in ${vcl_SOURCE_DIR}")
    add_library( vcl::vcl INTERFACE IMPORTED) # imported is visible only in current dir and below
    target_include_directories( vcl::vcl INTERFACE ${vcl_SOURCE_DIR})
    # Per default the dg library includes "vcl/vectorclass.h" but the vcl folder
    # does not exist in the vectorclass github, so tell dg to directly include "vectorclass.h"
    target_compile_definitions( vcl::vcl INTERFACE VCL_NO_INCLUDE_PREFIX)
endif()

target_link_libraries(dg_dg INTERFACE vcl::vcl)

# Add cccl

CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY "NVIDIA/cccl"
    VERSION 2.8.4
    SYSTEM ON
    EXCLUDE_FROM_ALL ON
)

target_link_libraries(dg_dg INTERFACE CCCL::CCCL)

if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
  # CUDA language must be enabled if device is CUDA
  target_link_libraries(dg_dg INTERFACE CUDA::cusparse)
  target_compile_features(dg_dg INTERFACE cuda_std_17)
endif()
# https://stackoverflow.com/questions/53256405/howto-pass-flag-to-nvcc-compiler-in-cmake
target_compile_options(dg_dg INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if( FELTOR_WITH_NCCL)
    target_compile_definitions( dg_dg INTERFACE DG_WITH_NCCL)
##### TODO Update once version 2.29 exists and nccl supports cmake
    #find_package( nccl)
    #target_link_libraries( dg_dg INTERFACE nccl::nccl)
##### In the meantime:
    # https://stackoverflow.com/questions/28597351/how-do-i-add-a-library-path-in-cmake
    add_library( nccl INTERFACE)
    add_library( nccl::nccl ALIAS nccl)
    target_include_directories( nccl INTERFACE "$ENV{NCCL_INC}")
    target_link_directories( nccl INTERFACE "$ENV{NCCL_LIB}")
    target_link_libraries( dg_dg INTERFACE nccl::nccl "-lnccl")
#####
endif()
if( FELTOR_CUDA_UNAWARE_MPI)
    target_compile_definitions( dg_dg INTERFACE DG_CUDA_UNAWARE_MPI)
endif()
# NVTX instrumentation: activates the dg/feltor NVTX ranges (feltor-integrate,
# feltor::Explicit::rhs, rhs-*, dg-symv-*, ...) guarded by DG_ENABLE_NVTX in
# inc/dg/backend/nvtx.h, so they show up as annotated ranges in Nsight Systems.
# The nvtx3 headers ship with the CUDA toolkit (header-only, nothing to link).
# Usage: cmake -DFELTOR_WITH_NVTX=ON ...
option( FELTOR_WITH_NVTX "Enable NVTX profiling ranges in dg and feltor (visible in Nsight Systems)." OFF)
if( FELTOR_WITH_NVTX)
    target_compile_definitions( dg_dg INTERFACE DG_ENABLE_NVTX)
endif()

# Use the single-reduction Chronopoulos-Gear PCG (dg::PCGmerged) inside
# MultigridCG2d elliptic solves instead of the stock dg::PCG.
# Usage: cmake -DFELTOR_DG_USE_PCG_MERGED=ON ...
option( FELTOR_DG_USE_PCG_MERGED "Use dg::PCGmerged (single-reduction CG) in MultigridCG2d." OFF)
if( FELTOR_DG_USE_PCG_MERGED)
    target_compile_definitions( dg_dg INTERFACE DG_USE_PCG_MERGED)
endif()
# NVSHMEM support: enables the production NVSHMEM halo-exchange path in symv.
# Requires NVSHMEM_HOME to be set in the environment.
# Usage: cmake -DFELTOR_WITH_NVSHMEM=ON ...
option( FELTOR_WITH_NVSHMEM "Enable NVSHMEM-backed halo exchange in symv" OFF)
if( FELTOR_WITH_NVSHMEM)
    # Locate the nvshmem install.  Search order:
    #   1. an explicit NVSHMEM_HOME (standalone nvshmem installs)
    #   2. the nvshmem bundled with nvhpc, which sets NVHPC_HOME; the libraries
    #      live under $NVHPC_HOME/comm_libs/<cudaver>/nvshmem (or a version-less
    #      comm_libs/nvshmem symlink).
    # The CMake link step runs nvcc as a separate command that does NOT inherit
    # nvcc's implicit nvshmem search path, so we resolve absolute paths here.
    set( _nvshmem_hints "")
    if( DEFINED ENV{NVSHMEM_HOME} AND NOT "$ENV{NVSHMEM_HOME}" STREQUAL "")
        list( APPEND _nvshmem_hints "$ENV{NVSHMEM_HOME}")
    endif()
    if( DEFINED ENV{NVHPC_HOME} AND NOT "$ENV{NVHPC_HOME}" STREQUAL "")
        # nvhpc ships one nvshmem per CUDA version under comm_libs/<major>.<minor>.
        # NVSHMEM and cudart MUST come from the same CUDA version: a 12.9 nvshmem
        # links cudaGetDeviceProperties_v2, which a 13.0 cudart no longer exports.
        # So prefer the comm_libs dir matching the CUDAToolkit we actually link
        # (found in the top-level CMakeLists) before falling back to a wildcard --
        # the wildcard alone sorts "12.9" before "13.0" and would mis-match.
        if( DEFINED CUDAToolkit_VERSION_MAJOR AND DEFINED CUDAToolkit_VERSION_MINOR)
            file( GLOB _nvhpc_nvshmem_match
                "$ENV{NVHPC_HOME}/comm_libs/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/nvshmem"
                "$ENV{NVHPC_HOME}/*/*/comm_libs/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/nvshmem")
            list( APPEND _nvshmem_hints ${_nvhpc_nvshmem_match})
        endif()
        file( GLOB _nvhpc_nvshmem
            "$ENV{NVHPC_HOME}/comm_libs/nvshmem"
            "$ENV{NVHPC_HOME}/comm_libs/*/nvshmem"
            "$ENV{NVHPC_HOME}/*/*/comm_libs/nvshmem"
            "$ENV{NVHPC_HOME}/*/*/comm_libs/*/nvshmem")
        list( APPEND _nvshmem_hints ${_nvhpc_nvshmem})
    endif()

    find_path(    NVSHMEM_INCLUDE_DIR    nvshmem.h     HINTS ${_nvshmem_hints} PATH_SUFFIXES include)
    find_library( NVSHMEM_HOST_LIBRARY   nvshmem_host  HINTS ${_nvshmem_hints} PATH_SUFFIXES lib lib64)
    find_library( NVSHMEM_DEVICE_LIBRARY nvshmem       HINTS ${_nvshmem_hints} PATH_SUFFIXES lib lib64)

    if( NOT NVSHMEM_INCLUDE_DIR OR NOT NVSHMEM_HOST_LIBRARY OR NOT NVSHMEM_DEVICE_LIBRARY)
        message( FATAL_ERROR
            "FELTOR_WITH_NVSHMEM=ON but nvshmem could not be located.\n"
            "Load an nvhpc module (which sets NVHPC_HOME, bundling nvshmem under\n"
            "$NVHPC_HOME/comm_libs/<cudaver>/nvshmem), or set NVSHMEM_HOME to a\n"
            "standalone install root containing include/ and lib/.\n"
            "  NVHPC_HOME            = $ENV{NVHPC_HOME}\n"
            "  NVSHMEM_INCLUDE_DIR   = ${NVSHMEM_INCLUDE_DIR}\n"
            "  NVSHMEM_HOST_LIBRARY  = ${NVSHMEM_HOST_LIBRARY}\n"
            "  NVSHMEM_DEVICE_LIBRARY= ${NVSHMEM_DEVICE_LIBRARY}")
    endif()
    message( STATUS "Found NVSHMEM host lib: ${NVSHMEM_HOST_LIBRARY}")

    # Even host-only programs must link the static device archive: on the
    # nvhpc-bundled NVSHMEM the init/finalize entry points (nvshmemi_init_thread,
    # nvshmemi_finalize) are defined ONLY in libnvshmem.a, not in
    # libnvshmem_host.so.  libnvshmem.a is relocatable device code, so its
    # __cudaRegisterLinkedBinary symbol requires a CUDA device-link (nvcc -dlink)
    # which CMake performs only for targets with CUDA_SEPARABLE_COMPILATION ON --
    # hence the global setting below.  Wrap the archive as an IMPORTED STATIC
    # target so CMake includes it in the device-link step.
    add_library( nvshmem_device STATIC IMPORTED)
    set_target_properties( nvshmem_device PROPERTIES
        IMPORTED_LOCATION "${NVSHMEM_DEVICE_LIBRARY}")

    add_library( nvshmem_feltor INTERFACE)
    target_include_directories( nvshmem_feltor INTERFACE "${NVSHMEM_INCLUDE_DIR}")
    # Link ONLY the monolithic static archive libnvshmem.a for the host side --
    # do NOT also link libnvshmem_host.so.  On this nvhpc-25.11 bundle the two
    # overlap: nvshmemx_putmem_on_stream (and its internal runtime state) is
    # defined in BOTH, while the bootstrap core (nvshmemi_common_init,
    # nvshmemi_boot*) lives ONLY in libnvshmem.a.  Linking both gives two copies
    # of the host state: nvshmem init runs through libnvshmem.a's common_init and
    # sets ITS state, but the gather's nvshmemx_putmem_on_stream resolves to
    # libnvshmem_host.so's copy, which is never initialised -> the put aborts with
    # "NVSHMEM API called before NVSHMEM initialization has completed" even though
    # init (and team setup) completed.  libnvshmem.a is self-contained (init +
    # host API + device runtime), so one archive = one consistent state.
    # cudart/cuda are referenced by the archive; --start-group lets the linker
    # iterate until the cross-references resolve regardless of emit order.
    target_link_libraries( nvshmem_feltor INTERFACE
        "-Wl,--start-group" nvshmem_device CUDA::cudart "-Wl,--end-group"
        "-lcuda")

    # CRITICAL: libnvshmem.a carries relocatable device code whose
    # __cudaRegisterLinkedBinary constructor registers NVSHMEM's device kernels
    # with the CUDA runtime.  An IMPORTED STATIC archive added through
    # target_link_libraries reaches only the HOST link (linkLibs.rsp); CMake does
    # NOT forward it to the `nvcc -dlink` step (dlink.txt).  When the archive is
    # absent from the device link, the device image is never registered, so the
    # first device/stream API (nvshmemx_putmem_on_stream) aborts with
    # "NVSHMEM API called before NVSHMEM initialization has completed" even though
    # host-side init (malloc, teams) succeeded.  Force the archive into the device
    # link explicitly via the DEVICE_LINK generator expression.
    target_link_options( nvshmem_feltor INTERFACE
        "$<DEVICE_LINK:${NVSHMEM_DEVICE_LIBRARY}>")

    # Required so nvcc compiles consumers with -rdc=true and runs the device-link
    # that resolves libnvshmem.a's __cudaRegisterLinkedBinary.  Set as a directory
    # variable here (before the benchmark/test targets are created below) so those
    # targets pick it up.  Only the NVSHMEM build tree configures this block.
    set( CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Propagate definition and link to all consumers of dg_dg
    target_compile_definitions( dg_dg INTERFACE DG_WITH_NVSHMEM)
    if( FELTOR_WITH_MPI)
        target_link_libraries( dg_dg INTERFACE nvshmem_feltor)
    endif()
endif()


# MW: cccl already links OpenMP
# if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "OMP")
#   find_package(OpenMP REQUIRED)
#   target_link_libraries(dg_dg INTERFACE OpenMP::OpenMP_CXX)
# endif()

# =============================================
# Set up benchmark and test targets
# MW a custom target is always out of date and has no output file
if(FELTOR_BUILD_BENCHMARKS)
  add_custom_target(dg_benchmarks)
endif()
if(FELTOR_BUILD_TESTS)
  add_custom_target(dg_tests)
endif()

# ============================================
# Add file utilities
# Defines the dg_file target

add_subdirectory(file)

# ============================================
# Add geometries utilities
# Defines the dg_geometries target

add_subdirectory(geometries)

# ============================================
# Add matrix utilities
# Defines the dg_matrix target

if(FELTOR_DG_WITH_MATRIX)
    add_subdirectory(matrix)
endif()


# ============================================
# Add dg benchmarks

if(FELTOR_BUILD_BENCHMARKS)
    include(FeltorBenchmarks)
    # There are no backend and topology benchmarks
    # All benchmarks can be compiled for MPI
    foreach(benchmark_path IN LISTS DG_BENCHMARKS)
        add_dg_benchmark("${benchmark_path}" "" ${FELTOR_WITH_MPI} target_name )
    endforeach()
    # NVSHMEM is exercised through mpi_matrix_b (already in DG_BENCHMARKS): when
    # FELTOR_WITH_NVSHMEM=ON the production symv path dispatches to NVSHMEM via
    # DG_WITH_NVSHMEM, so no separate benchmark source is required.
endif()

# ============================================
# Add dg tests
# We need to parse the paths to be sure to put topology and backend tests in the right folder

if(FELTOR_BUILD_TESTS)
    include(FeltorTests)
    if(FELTOR_WITH_MPI)
        foreach(test_path IN LISTS DG_MPI_TESTS)
            cmake_path( REMOVE_FILENAME test_path OUTPUT_VARIABLE outvar)
            if( outvar STREQUAL "topology/")
                add_dg_test("${test_path}" "topology" ON target_name )
            elseif( outvar STREQUAL "backend/")
                add_dg_test("${test_path}" "backend" ON target_name )
            else()
                add_dg_test("${test_path}" "" ON target_name )
            endif()
        endforeach()
    else()
        foreach(test_path IN LISTS DG_TESTS)
            cmake_path( REMOVE_FILENAME test_path OUTPUT_VARIABLE outvar)
            if( outvar STREQUAL "topology/")
                add_dg_test("${test_path}" "topology" OFF target_name )
            elseif( outvar STREQUAL "backend/")
                add_dg_test("${test_path}" "backend" OFF target_name )
            else()
                add_dg_test("${test_path}" "" OFF target_name )
            endif()
        endforeach()
    endif()
endif()

# ============================================
# Add Install rules
if(FELTOR_INSTALL)

    # cmake --install build --prefix install-dir

    # This install rule copies the FILE_SET "dg_headers" to install-dir/include/ (i.e. results in include/dg/...)
    # and associates the target to an export set and a install component set
    install(
        TARGETS dg_dg COMPONENT feltor_dg
        EXPORT feltor_dg # associate installed target files with an export called "feltor_dg"
        FILE_SET dg_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # This install rule creates a file ("feltor_dg-targets.cmake") at the specified destination
    # that can (theoretically) be included by other projects and they have all targets (in feltor_dg) available
    # Usually, this is an intermediate step towards making "find_package" work for our project

# ATTENTION: This export currently (3.6.25) does not work because our external
# dependencies are handled by CPM which is just a wrapper around FetchContent,
# which itself is as if we add_subdirectorie'd the external packages. This
# means e.g. the target nlohman_json::nlohmann_json is not an IMPORTED target
# in the same way that find_package makes it. Cmake will thus complain that the
# target is "not in any export set".  In order for this to work every
# dependency has to be find_package'd and an external package manager has to be
# used to install these dependencies on the system Now, find_package has its
# own nuances and currently does not work e.g. for VCL
    install(
        EXPORT feltor_dg # all targets within this set
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/feltor"
        NAMESPACE feltor::dg::
        FILE feltor_dg-targets.cmake
        COMPONENT feltor_dg
    )
endif()

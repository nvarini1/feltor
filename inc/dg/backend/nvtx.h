#pragma once
// Lightweight NVTX (NVIDIA Tools Extension) instrumentation helpers for dg.
//
// NVTX ranges/markers show up as annotated time ranges in NVIDIA Nsight
// Systems (nsys) profiles, making it easy to see e.g. how communication and
// computation overlap in the distributed symv.
//
// Instrumentation is *opt-in* and *zero cost* unless enabled: compile with
//     -DDG_ENABLE_NVTX
// The nvtx3 C API used here is header-only (shipped with the CUDA Toolkit /
// nvhpc), so no additional library needs to be linked.
//
// Usage:
//     #include "backend/nvtx.h"
//     void foo() {
//         DG_NVTX_RANGE("foo");            // scoped range, auto-closed at }
//         { DG_NVTX_RANGE("phase-1"); ... }
//     }
// or manual push/pop across non-scoped boundaries:
//     DG_NVTX_PUSH("phase"); ...; DG_NVTX_POP();
//
///@cond

#if defined(DG_ENABLE_NVTX)

#include <nvtx3/nvToolsExt.h>
#include <cstdint>

namespace dg
{
namespace nvtx
{
// RAII range: pushes on construction, pops on destruction.
class Range
{
  public:
    explicit Range( const char* name) { nvtxRangePushA( name); }
    Range( const char* name, uint32_t argb_color)
    {
        nvtxEventAttributes_t attr = {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = argb_color;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        nvtxRangePushEx( &attr);
    }
    ~Range() { nvtxRangePop(); }
    Range( const Range&) = delete;
    Range& operator=( const Range&) = delete;
};
} // namespace nvtx
} // namespace dg

#define DG_NVTX_CONCAT_(a,b) a##b
#define DG_NVTX_CONCAT(a,b) DG_NVTX_CONCAT_(a,b)

// Scoped range named for the enclosing block; auto-closed at end of scope.
#define DG_NVTX_RANGE(name) \
    dg::nvtx::Range DG_NVTX_CONCAT(dg_nvtx_range_,__LINE__)(name)
// Scoped colored range. color is 0xAARRGGBB.
#define DG_NVTX_RANGE_C(name,color) \
    dg::nvtx::Range DG_NVTX_CONCAT(dg_nvtx_range_,__LINE__)(name,color)
// Manual push / pop (must be balanced) and instantaneous marker.
#define DG_NVTX_PUSH(name) nvtxRangePushA(name)
#define DG_NVTX_POP()      nvtxRangePop()
#define DG_NVTX_MARK(name) nvtxMarkA(name)

#else // DG_ENABLE_NVTX not defined -> all no-ops

#define DG_NVTX_RANGE(name)        do {} while(0)
#define DG_NVTX_RANGE_C(name,color) do {} while(0)
#define DG_NVTX_PUSH(name)         do {} while(0)
#define DG_NVTX_POP()              do {} while(0)
#define DG_NVTX_MARK(name)         do {} while(0)

#endif // DG_ENABLE_NVTX

///@endcond

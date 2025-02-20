set(PKG_NAME "OVXDRV")

message("include driver sdk from ${EXTERNAL_VIV_SDK}")
set(OVXDRV_INCLUDE_DIRS)
list(APPEND OVXDRV_INCLUDE_DIRS
    ${EXTERNAL_VIV_SDK}/include
    ${EXTERNAL_VIV_SDK}/include/CL)

if("${CONFIG}" STREQUAL "BUILDROOT")
    set(VIV_SDK_DRIVER_PREFIX "usr/lib")
else()
    set(VIV_SDK_DRIVER_PREFIX "drivers")
endif()

message("using driver libs from ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}")

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    ${EXTERNAL_VIV_SDK}/lib/armv7a-linux-baseline/libebg_utils.so
    ${EXTERNAL_VIV_SDK}/lib/armv7a-linux-baseline/libsynapnb.so)
#    ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}/libGAL.so
#    ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}/libOpenVX.so
#    ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}/libOpenVXU.so
#    ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}/libVSC.so
#    ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}/libArchModelSw.so
#    ${EXTERNAL_VIV_SDK}/${VIV_SDK_DRIVER_PREFIX}/libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})

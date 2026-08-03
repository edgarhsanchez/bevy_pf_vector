//! rive-runtime C ABI surface (feature = "rive-native").
//!
//! Deliberately minimal: RenderContext lifecycle, frame begin/flush, and target
//! binding. Path/paint construction stays on the C++ side — we are binding the
//! renderer, not reimplementing the .riv format.
//!
//! NOTE: rive-runtime does not ship a stable C header for this. A thin C++
//! shim in `cpp/shim.cpp` compiled via `cc` is required. Not yet written.

#[repr(C)]
pub struct RiveRenderContext {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    /// Construct a RenderContext against an already-created Vulkan device.
    /// The device, queue, and physical device are borrowed from Bevy via
    /// wgpu-hal `Device::as_hal::<Vulkan>()` and must outlive the context.
    pub fn pf_rive_ctx_new_vk(
        instance: *mut core::ffi::c_void,
        physical_device: *mut core::ffi::c_void,
        device: *mut core::ffi::c_void,
        queue: *mut core::ffi::c_void,
        queue_family_index: u32,
    ) -> *mut RiveRenderContext;

    pub fn pf_rive_ctx_free(ctx: *mut RiveRenderContext);

    /// Bind a Bevy-owned VkImage as the render target. No copy.
    pub fn pf_rive_bind_target(
        ctx: *mut RiveRenderContext,
        image: *mut core::ffi::c_void,
        width: u32,
        height: u32,
        format: u32,
    );

    pub fn pf_rive_begin_frame(ctx: *mut RiveRenderContext, clear: u32);
    pub fn pf_rive_flush(ctx: *mut RiveRenderContext);
}

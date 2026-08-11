//! Low-level plumbing shared by every binding module: library loading, the
//! packed-call macro, argument/return conversion helpers and the opaque object
//! wrappers around the C++ types.
//!
//! Everything in here is `pub(crate)`; the public API never exposes tvm-ffi
//! types directly.

pub(crate) mod handle;
pub(crate) mod reflection;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tvm_ffi::{Any, Function, Module};

use crate::error::{Error, Result};

/// A type key that is registered iff the xgrammar bindings library has been
/// loaded into this process.
const CANARY_TYPE_KEY: &str = "xgrammar.tvm_ffi_binding.Grammar";

/// Whether the xgrammar registrations are present in this process.
fn bindings_present() -> bool {
    reflection::type_key_registered(CANARY_TYPE_KEY)
}

/// Environment variable pointing at `libxgrammar_bindings`.
const LIB_ENV: &str = "XGRAMMAR_BINDINGS_LIB";

/// The platform-specific file name of the bindings library
/// (`libxgrammar_bindings.so` / `.dylib` / `xgrammar_bindings.dll`).
///
/// Useful with [`load_library`] when loading the library from a known
/// directory, e.g. a build tree.
#[cfg(target_os = "macos")]
pub const BINDINGS_LIBRARY_FILENAME: &str = "libxgrammar_bindings.dylib";
/// The platform-specific file name of the bindings library.
#[cfg(target_os = "windows")]
pub const BINDINGS_LIBRARY_FILENAME: &str = "xgrammar_bindings.dll";
/// The platform-specific file name of the bindings library
/// (`libxgrammar_bindings.so` / `.dylib` / `xgrammar_bindings.dll`).
///
/// Useful with [`load_library`] when loading the library from a known
/// directory, e.g. a build tree.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub const BINDINGS_LIBRARY_FILENAME: &str = "libxgrammar_bindings.so";

const LIB_BASENAME: &str = BINDINGS_LIBRARY_FILENAME;

static LOADED: AtomicBool = AtomicBool::new(false);
static LOAD_LOCK: Mutex<()> = Mutex::new(());

/// Load the xgrammar bindings library from an explicit path.
///
/// Calling this is only required when the library cannot be found
/// automatically. The default search order on first use is: registrations
/// already present in the process, then the path in `$XGRAMMAR_BINDINGS_LIB`,
/// then the platform library name via the system loader search path. The
/// library stays loaded for the lifetime of the process. Idempotent: once a
/// library has been loaded successfully, later calls are no-ops.
pub fn load_library(path: impl AsRef<str>) -> Result<()> {
    let _guard = LOAD_LOCK.lock().unwrap();
    if LOADED.load(Ordering::Acquire) {
        return Ok(());
    }
    try_load(path.as_ref())?;
    LOADED.store(true, Ordering::Release);
    Ok(())
}

/// Make sure the bindings are available, loading the library on first use.
///
/// Search order:
/// 1. registrations already present in the process (e.g. the host application
///    or an embedded Python interpreter already loaded the library);
/// 2. the path in `$XGRAMMAR_BINDINGS_LIB`;
/// 3. the default candidates: the repository checkout the crate was built
///    from, the `xgrammar` Python package discovered at build time, and the
///    platform library name via the system loader search path.
pub(crate) fn ensure_loaded() -> Result<()> {
    if LOADED.load(Ordering::Acquire) {
        return Ok(());
    }
    let _guard = LOAD_LOCK.lock().unwrap();
    if LOADED.load(Ordering::Acquire) {
        return Ok(());
    }
    if bindings_present() {
        LOADED.store(true, Ordering::Release);
        return Ok(());
    }
    let attempt = if let Ok(path) = std::env::var(LIB_ENV) {
        try_load(&path)
    } else {
        try_load_default_candidates()
    };
    match attempt {
        Ok(()) => {
            LOADED.store(true, Ordering::Release);
            Ok(())
        }
        Err(err) => Err(Error::Library(format!(
            "cannot load the xgrammar bindings library: {err}. Set {LIB_ENV} to the \
             path of {LIB_BASENAME} (built by the CMake target `xgrammar_bindings`, \
             or shipped inside the `xgrammar` Python wheel), or call \
             xgrammar::load_library() before any other API."
        ))),
    }
}

/// Default load candidates, tried in order (missing ones are skipped):
/// 1. the repository checkout this crate was built from (development builds);
/// 2. the `xgrammar` Python package discovered at build time (pip installs);
/// 3. the platform library name via the system loader search path.
fn try_load_default_candidates() -> Result<()> {
    let dev_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../python/xgrammar")
        .join(BINDINGS_LIBRARY_FILENAME);
    let python_package_dir = env!("XGRAMMAR_PYTHON_PACKAGE_DIR");

    let mut candidates = vec![dev_path.to_string_lossy().into_owned()];
    if !python_package_dir.is_empty() {
        candidates.push(
            std::path::Path::new(python_package_dir)
                .join(BINDINGS_LIBRARY_FILENAME)
                .to_string_lossy()
                .into_owned(),
        );
    }
    candidates.push(LIB_BASENAME.to_string());

    let mut last_err = None;
    for candidate in &candidates {
        match try_load(candidate) {
            Ok(()) => return Ok(()),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.expect("at least one candidate is always tried"))
}

/// dlopen `path`, keep it alive forever, and verify the registrations are
/// there.
fn try_load(path: &str) -> Result<()> {
    let module =
        Module::load_from_file(path).map_err(|e| Error::Library(e.message().to_string()))?;
    // Dropping the last `Module` handle dlcloses the library and would leave
    // dangling registrations behind; the bindings must stay for the lifetime
    // of the process.
    std::mem::forget(module);
    if !bindings_present() {
        return Err(Error::Library(format!(
            "{path} was loaded but does not register the xgrammar types; it does not look \
             like an xgrammar bindings library"
        )));
    }
    Ok(())
}

/// Resolve an FFI function by name and cache it per thread.
///
/// Names are the global-function names when they exist (the `testing.*`,
/// `kernels.*` and `config.*` functions, plus tvm-ffi's own `ffi.*` helpers);
/// everything else is a `"<type_key>.<method_name>"` reflected type method,
/// resolved through [`reflection`]. `Function` is not `Sync`, so the cache is
/// thread-local; lookups after the first are a `OnceCell` read.
pub(crate) fn cached_global(
    cell: &'static std::thread::LocalKey<std::cell::OnceCell<Function>>,
    name: &str,
) -> Result<Function> {
    ensure_loaded()?;
    cell.with(|c| {
        if let Some(f) = c.get() {
            return Ok(f.clone());
        }
        let f = match Function::get_global(name) {
            Ok(f) => f,
            Err(global_err) => {
                reflection::reflected_method(name).map_err(|_| Error::from_ffi(global_err))?
            }
        };
        let _ = c.set(f.clone());
        Ok(f)
    })
}

/// Call a global function with the given arguments.
///
/// Each expansion owns a per-thread `OnceCell<Function>`, so the registry is
/// consulted once per thread per call site. Arguments must implement
/// `tvm_ffi::AnyCompatible` (pass `ffi::none()` for optional slots).
/// Evaluates to `Result<tvm_ffi::Any, Error>`.
macro_rules! ffi_call {
    ($name:expr $(, $arg:expr)* $(,)?) => {{
        thread_local! {
            static FUNC: std::cell::OnceCell<tvm_ffi::Function> =
                const { std::cell::OnceCell::new() };
        }
        match $crate::ffi::cached_global(&FUNC, $name) {
            Ok(func) => func
                .call_packed(&[$(tvm_ffi::AnyView::from(&$arg)),*])
                .map_err($crate::error::Error::from_ffi),
            Err(err) => Err(err),
        }
    }};
}
pub(crate) use ffi_call;

/// Convert a returned `Any` into `T`, mapping conversion failures.
pub(crate) fn ret<T>(any: Any) -> Result<T>
where
    T: TryFrom<Any, Error = tvm_ffi::Error>,
{
    T::try_from(any).map_err(Error::from_ffi)
}

/// An optional FFI argument: `Some(v)` becomes the value, `None` becomes None.
pub(crate) fn opt_any<T: tvm_ffi::AnyCompatible>(value: Option<T>) -> Any {
    match value {
        Some(v) => Any::from(v),
        None => Any::new(),
    }
}

/// A borrowed DLTensor argument. Keeps the `DLTensor` struct and its shape
/// alive for as long as the argument is in scope; the tensor data itself is
/// borrowed from the caller.
pub(crate) struct DlArg {
    _shape: Box<[i64]>,
    _tensor: Box<tvm_ffi::tvm_ffi_sys::dlpack::DLTensor>,
    any: Any,
}

impl DlArg {
    /// # Safety
    ///
    /// `data` must point to memory matching `shape`/`code`/`bits`, valid (and
    /// exclusively borrowed if the callee writes) until the `DlArg` is dropped.
    unsafe fn new(data: *mut std::ffi::c_void, shape: &[i64], code: u8, bits: u8) -> Self {
        use tvm_ffi::tvm_ffi_sys::dlpack::{DLDataType, DLDevice, DLDeviceType, DLTensor};
        let shape_box: Box<[i64]> = shape.into();
        let mut tensor = Box::new(DLTensor {
            data,
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: shape.len() as i32,
            dtype: DLDataType {
                code,
                bits,
                lanes: 1,
            },
            shape: shape_box.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        });
        let mut raw = tvm_ffi::tvm_ffi_sys::TVMFFIAny::new();
        raw.type_index = tvm_ffi::TypeIndex::kTVMFFIDLTensorPtr as i32;
        raw.data_union.v_ptr = tensor.as_mut() as *mut DLTensor as *mut std::ffi::c_void;
        // A DLTensorPtr payload is plain data: dropping the Any is a no-op.
        let any = Any::from_raw_ffi_any(raw);
        Self {
            _shape: shape_box,
            _tensor: tensor,
            any,
        }
    }

    /// A writable int32 tensor over `data` with the given shape.
    pub(crate) fn int32(data: &mut [i32], shape: &[i64]) -> Self {
        debug_assert_eq!(shape.iter().product::<i64>() as usize, data.len());
        unsafe { Self::new(data.as_mut_ptr() as *mut _, shape, 0, 32) }
    }

    /// A read-only 1-D int64 tensor over `data` (the FFI signature still
    /// takes a mutable pointer, but the callee never writes).
    pub(crate) fn int64_readonly(data: &[i64]) -> Self {
        unsafe { Self::new(data.as_ptr() as *mut _, &[data.len() as i64], 0, 64) }
    }

    /// A writable 1-D float32 tensor over `data`.
    pub(crate) fn float32(data: &mut [f32]) -> Self {
        unsafe { Self::new(data.as_mut_ptr() as *mut _, &[data.len() as i64], 2, 32) }
    }

    /// The `Any` to pass as the FFI argument.
    pub(crate) fn any(&self) -> &Any {
        &self.any
    }
}

/// Build a heterogeneous `ffi::Array` from AnyViews.
///
/// At the pinned tvm-ffi, `Array<Any>` is not constructible from Rust (`Any`
/// does not implement `AnyCompatible`), so mixed-type arrays go through the
/// registered `ffi.Array` constructor instead.
pub(crate) fn mixed_array(views: &[tvm_ffi::AnyView]) -> Result<Any> {
    thread_local! {
        static FUNC: std::cell::OnceCell<Function> = const { std::cell::OnceCell::new() };
    }
    cached_global(&FUNC, "ffi.Array")?
        .call_packed(views)
        .map_err(Error::from_ffi)
}

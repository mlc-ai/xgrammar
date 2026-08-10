/*
 * Copyright (c) 2024 by Contributors
 * \file tvm_ffi_loader.c
 * \brief Runtime loader shim for the TVM FFI C ABI used by the Rust binding.
 */

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Do not import/export these declarations: the shim only needs them for the
// final static link. Keeping them out of the executable's dynamic symbol table
// prevents interposition while the real runtime executes its initializers.
#define TVM_FFI_DLL
#define TVM_FFI_DLL_EXPORT
#include <tvm/ffi/c_api.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif

#ifndef TVM_FFI_LIBRARY_PATH
#error "TVM_FFI_LIBRARY_PATH must be supplied by build.rs"
#endif

#ifndef TVM_FFI_LIBRARY_BASENAME
#error "TVM_FFI_LIBRARY_BASENAME must be supplied by build.rs"
#endif

typedef int (*TVMFFITypeKeyToIndexFn)(const TVMFFIByteArray*, int32_t*);
typedef int (*TVMFFIFunctionGetGlobalFn)(const TVMFFIByteArray*, TVMFFIObjectHandle*);
typedef void (*TVMFFIErrorMoveFromRaisedFn)(TVMFFIObjectHandle*);
typedef int (*TVMFFIErrorCreateFn)(const TVMFFIByteArray*, const TVMFFIByteArray*, const TVMFFIByteArray*, TVMFFIObjectHandle*);
typedef const TVMFFITypeInfo* (*TVMFFIGetTypeInfoFn)(int32_t);

typedef struct {
  TVMFFITypeKeyToIndexFn type_key_to_index;
  TVMFFIFunctionGetGlobalFn function_get_global;
  TVMFFIErrorMoveFromRaisedFn error_move_from_raised;
  TVMFFIErrorCreateFn error_create;
  TVMFFIGetTypeInfoFn get_type_info;
} TVMFFIRustAPI;

static TVMFFIRustAPI g_api;

#ifdef _WIN32

static HMODULE g_library;
static INIT_ONCE g_init_once = INIT_ONCE_STATIC_INIT;
typedef FARPROC TVMFFISymbol;

static TVMFFISymbol LoadSymbol(const char* name) {
  FARPROC symbol = GetProcAddress(g_library, name);
  if (symbol == NULL) {
    fprintf(
        stderr,
        "xgrammar: cannot resolve %s from the TVM FFI runtime (error %lu)\n",
        name,
        (unsigned long)GetLastError()
    );
    abort();
  }
  return symbol;
}

static BOOL CALLBACK InitializeTVMFFI(PINIT_ONCE once, PVOID parameter, PVOID* context) {
  (void)once;
  (void)parameter;
  (void)context;
  const char* override_path = getenv("TVM_FFI_LIBRARY_PATH");
  if (override_path != NULL && override_path[0] != '\0') {
    g_library = LoadLibraryA(override_path);
  }
  if (g_library == NULL) {
    g_library = LoadLibraryA(TVM_FFI_LIBRARY_PATH);
  }
  if (g_library == NULL) {
    g_library = LoadLibraryA(TVM_FFI_LIBRARY_BASENAME);
  }
  if (g_library == NULL) {
    fprintf(
        stderr,
        "xgrammar: cannot load the TVM FFI runtime; tried TVM_FFI_LIBRARY_PATH, %s, and %s "
        "(error %lu)\n",
        TVM_FFI_LIBRARY_PATH,
        TVM_FFI_LIBRARY_BASENAME,
        (unsigned long)GetLastError()
    );
    abort();
  }
#define LOAD_FIELD(field, symbol)                        \
  do {                                                   \
    TVMFFISymbol address = LoadSymbol(#symbol);          \
    if (sizeof(g_api.field) != sizeof(address)) abort(); \
    memcpy(&g_api.field, &address, sizeof(address));     \
  } while (0)
  LOAD_FIELD(type_key_to_index, TVMFFITypeKeyToIndex);
  LOAD_FIELD(function_get_global, TVMFFIFunctionGetGlobal);
  LOAD_FIELD(error_move_from_raised, TVMFFIErrorMoveFromRaised);
  LOAD_FIELD(error_create, TVMFFIErrorCreate);
  LOAD_FIELD(get_type_info, TVMFFIGetTypeInfo);
#undef LOAD_FIELD
  return TRUE;
}

static void EnsureTVMFFILoaded(void) {
  if (!InitOnceExecuteOnce(&g_init_once, InitializeTVMFFI, NULL, NULL)) {
    fprintf(
        stderr,
        "xgrammar: cannot initialize the TVM FFI runtime loader (error %lu)\n",
        (unsigned long)GetLastError()
    );
    abort();
  }
}

#else

static void* g_library;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
typedef void* TVMFFISymbol;

static TVMFFISymbol LoadSymbol(const char* name) {
  dlerror();
  void* symbol = dlsym(g_library, name);
  const char* error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "xgrammar: cannot resolve %s from the TVM FFI runtime: %s\n", name, error);
    abort();
  }
  return symbol;
}

static void InitializeTVMFFI(void) {
  const char* override_path = getenv("TVM_FFI_LIBRARY_PATH");
  if (override_path != NULL && override_path[0] != '\0') {
    g_library = dlopen(override_path, RTLD_NOW | RTLD_GLOBAL);
  }
  if (g_library == NULL) {
    g_library = dlopen(TVM_FFI_LIBRARY_PATH, RTLD_NOW | RTLD_GLOBAL);
  }
  if (g_library == NULL) {
    g_library = dlopen(TVM_FFI_LIBRARY_BASENAME, RTLD_NOW | RTLD_GLOBAL);
  }
  if (g_library == NULL) {
    fprintf(
        stderr,
        "xgrammar: cannot load the TVM FFI runtime; tried TVM_FFI_LIBRARY_PATH, %s, and %s: "
        "%s\n",
        TVM_FFI_LIBRARY_PATH,
        TVM_FFI_LIBRARY_BASENAME,
        dlerror()
    );
    abort();
  }
#define LOAD_FIELD(field, symbol)                        \
  do {                                                   \
    TVMFFISymbol address = LoadSymbol(#symbol);          \
    if (sizeof(g_api.field) != sizeof(address)) abort(); \
    memcpy(&g_api.field, &address, sizeof(address));     \
  } while (0)
  LOAD_FIELD(type_key_to_index, TVMFFITypeKeyToIndex);
  LOAD_FIELD(function_get_global, TVMFFIFunctionGetGlobal);
  LOAD_FIELD(error_move_from_raised, TVMFFIErrorMoveFromRaised);
  LOAD_FIELD(error_create, TVMFFIErrorCreate);
  LOAD_FIELD(get_type_info, TVMFFIGetTypeInfo);
#undef LOAD_FIELD
}

static void EnsureTVMFFILoaded(void) {
  int status = pthread_once(&g_init_once, InitializeTVMFFI);
  if (status != 0) {
    fprintf(stderr, "xgrammar: cannot initialize the TVM FFI runtime loader (error %d)\n", status);
    abort();
  }
}

#endif

int TVMFFITypeKeyToIndex(const TVMFFIByteArray* type_key, int32_t* out_tindex) {
  EnsureTVMFFILoaded();
  return g_api.type_key_to_index(type_key, out_tindex);
}

int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out) {
  EnsureTVMFFILoaded();
  return g_api.function_get_global(name, out);
}

void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result) {
  EnsureTVMFFILoaded();
  g_api.error_move_from_raised(result);
}

int TVMFFIErrorCreate(
    const TVMFFIByteArray* kind,
    const TVMFFIByteArray* message,
    const TVMFFIByteArray* backtrace,
    TVMFFIObjectHandle* out
) {
  EnsureTVMFFILoaded();
  return g_api.error_create(kind, message, backtrace, out);
}

const TVMFFITypeInfo* TVMFFIGetTypeInfo(int32_t type_index) {
  EnsureTVMFFILoaded();
  return g_api.get_type_info(type_index);
}

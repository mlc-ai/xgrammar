option("XGRAMMAR_BUILD_PYTHON_BINDINGS", {
    default = false, description = "Build Python bindings"
})
option("XGRAMMAR_BUILD_CXX_TESTS", {
    default = false, description = "Build C++ tests"
})

add_requires("dlpack 1.1", "rapidjson 2025.02.05")
add_rules("mode.debug", "mode.release")

-- config dependencies
if has_config("XGRAMMAR_BUILD_PYTHON_BINDINGS") then
    add_requires("nanobind v2.5.0")
end
if has_config("XGRAMMAR_BUILD_CXX_TESTS") then
    add_requires("gtest 1.16.0", {configs = {main = true}})
end

-- common global settings
set_languages("c++17")
add_includedirs("3rdparty/picojson")
add_includedirs("include", {public = true})
add_packages("dlpack", "rapidjson")
add_defines("RAPIDJSON_HAS_STDSTRING")
add_cxxflags("-Wno-unused-parameter")
set_warnings("all", "extra", "error")

-- xgrammar static library
target("xgrammar")
    set_kind("static")
    add_files("cpp/*.cc")
    add_files("cpp/support/*.cc")
    if has_config("XGRAMMAR_BUILD_PYTHON_BINDINGS") then
        add_files("cpp/nanobind/*.cc")
        add_packages("nanobind")
    end
target_end()

-- build xgrammar C++ gtests
if has_config("XGRAMMAR_BUILD_CXX_TESTS") then
    target("test")
        set_kind("binary")
        add_includedirs("cpp", {public = true})
        add_cxflags("-Wno-cpp")
        add_files("tests/cpp/*.cc")
        add_deps("xgrammar")
        add_packages("gtest")
        add_tests("default")
        -- this is a workaround for the gtest main function on Windows
        if is_plat("windows") then
            add_ldflags("/subsystem:console")
        end
    target_end()
end

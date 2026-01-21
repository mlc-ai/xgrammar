#[path = "build/mod.rs"]
mod build;

fn main() {
    build::diagnostics::check_requirements();

    #[cfg(target_os = "windows")]
    {
        build::windows::configure_libclang_early();

        if std::process::Command::new("link.exe")
            .arg("/?")
            .output()
            .is_err()
        {
            build::windows::print_path_setup_instructions();
        }
    }

    let ctx = build::common::collect_build_context();
    let destination_path = build::cmake::build_xgrammar_cmake(&ctx);
    build::cmake::link_xgrammar_static(&ctx, &destination_path);
    build::autocxx::build_autocxx_bridge(&ctx);
    build::autocxx::copy_headers_for_generated_rust_code(&ctx);
    build::autocxx::format_generated_bindings_optional(&ctx.out_dir);
    build::autocxx::strip_autocxx_generated_doc_comments(&ctx.out_dir);
}

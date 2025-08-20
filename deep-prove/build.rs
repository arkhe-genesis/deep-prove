use std::{env, fs, path::PathBuf};

use miette::IntoDiagnostic;
use protox::prost::Message;

fn main() -> miette::Result<()> {
    println!("cargo:rerun-if-changed=../.git");
    export_version_from_git();

    println!("cargo:rerun-if-changed=../lagrange-protobuf/");
    let file_descriptors = protox::compile(["proto/lagrange.proto"], ["../lagrange-protobuf/"])?;
    let file_descriptor_path = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"))
        .join("lagrange_descriptor.bin");

    fs::write(&file_descriptor_path, file_descriptors.encode_to_vec()).unwrap();

    tonic_build::configure()
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .extern_path(".google.protobuf.Duration", "::prost_wkt_types::Duration")
        .file_descriptor_set_path(file_descriptor_path)
        .build_server(true)
        .compile_fds(file_descriptors)
        .into_diagnostic()?;

    Ok(())
}

/// If git is available, exports compile-time env var `GIT_DESC_VER` with version based on git tag,
/// commits and uncommitted changes on top of tag.
///
/// This is used in conjunction with `deep_prove::get_version!()`.
///
/// # Example
///
/// ```
/// println!("cargo:rerun-if-changed=../.git");
/// let version = common_build::export_version_from_git();
/// ```
pub fn export_version_from_git() {
    if let Ok(output) = std::process::Command::new("git")
        .args(["describe", "--dirty"])
        .output()
    {
        let version =
            String::from_utf8(output.stdout).expect("git describe output must be UTF8 string");
        println!("cargo:rustc-env=GIT_DESC_VER={version}");
    }
}

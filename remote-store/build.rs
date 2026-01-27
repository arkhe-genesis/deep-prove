fn main() {
    println!("cargo:rerun-if-changed=../.git");
    export_version_from_git();
}

/// If git is available, exports compile-time env var `GIT_DESC_VER` with version based on git tag,
/// commits and uncommitted changes on top of tag.
///
/// This is used in conjunction with `remote_store::get_version!()`.
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

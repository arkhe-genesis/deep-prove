pub mod middleware;
pub mod store;

/// Get version from `GIT_DESC_VER` compile-time env var, which must be
/// provided by build.rs via `export_version_from_git`, with a
/// fallback to version from Cargo manifest if it's an empty string.
#[macro_export]
macro_rules! get_version {
    () => {{
        let git_desc_ver = option_env!("GIT_DESC_VER");
        match git_desc_ver {
            Some(version) if !version.trim().is_empty() => version,
            _ => clap::crate_version!(),
        }
    }};
}

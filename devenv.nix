{ pkgs, lib, config, inputs, ... }:

let
  # Combine NLTK data packages into a single directory
  nltk-data = pkgs.symlinkJoin {
    name = "nltk-data";
    paths = [
      pkgs.nltk-data.punkt
      pkgs.nltk-data.punkt-tab
    ];
  };
in
{
  cachix.enable = false;
  dotenv.enable = true;

  packages = [
    # General dev.
    pkgs.git pkgs.git-lfs pkgs.openssl pkgs.cmake pkgs.git-cliff
    pkgs.cudatoolkit

    # Rust crates build deps
    pkgs.openssl pkgs.llvmPackages.libclang.lib pkgs.protobuf
    pkgs.cargo-nextest

    # For tensor visualization
    pkgs.gnuplot_qt pkgs.d2

    # Python dependencies
    pkgs.zlib
    pkgs.openblas
  ] ++ (if pkgs.stdenv.isLinux then [ pkgs.linuxPackages.nvidia_x11 ] else []);

  env = {
    OPENSSL_DEV = pkgs.openssl.dev;
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
    LD_LIBRARY_PATH = "${pkgs.zlib}/lib:${pkgs.openblas}/lib:${pkgs.stdenv.cc.cc.lib}/lib"
                      + (if pkgs.stdenv.isLinux then " ${pkgs.linuxPackages.nvidia_x11}/lib" else "");
    # Point NLTK to pre-installed data in nix store
    NLTK_DATA = "${nltk-data}";
    OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318";
    OTEL_SERVICE_NAME="dp-worker";
  };

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
  python zkml/assets/scripts/llms/gpt2_internal.py --output-dir ./zkml/assets/scripts/llms/ --export-model
  cargo test --release -p zkml -- --test-threads 1
  '';

  languages.rust = {
    enable = true;
    channel = "nightly";
    version = "2026-01-27";
    mold.enable = pkgs.stdenv.isLinux;
    rustflags = "--cfg tokio_unstable"; # Needed for metrics
  };
  languages.python = {
    enable = true;
    venv.enable = true;
    # More recent torch ends up in SIGFPE when evaluating Gemma3
    venv.requirements = ''
    torch == 2.7.1
    torchmetrics
    torchvision
    '';
  };

  # https://devenv.sh/git-hooks/
  git-hooks.hooks = {
    # actionlint.enable = true;
    check-merge-conflicts.enable = true;
    ripsecrets.enable = true;
    rustfmt = {
      enable = true;
      settings.color = "auto";
    };
    black = {
      enable = true;
    };
    taplo = {
      enable = true;
    };
    typos = {
      enable = true;
      settings = {
        format = "brief";
        write = true;
        configPath = "typos.toml";
      };
    };
  };
}

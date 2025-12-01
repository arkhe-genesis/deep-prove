{ pkgs, lib, config, inputs, ... }:

{
  cachix.enable = false;

  packages = [
    # General dev.
    pkgs.git pkgs.git-lfs pkgs.openssl pkgs.cmake pkgs.git-cliff

    # Rust crates build deps
    pkgs.openssl pkgs.llvmPackages.libclang.lib pkgs.protobuf
    pkgs.cargo-nextest

    # For tensor visualization
    pkgs.gnuplot_qt pkgs.d2

    # Python dependencies
    pkgs.zlib
    pkgs.openblas
  ];

  env = {
    OPENSSL_DEV = pkgs.openssl.dev;
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
    LD_LIBRARY_PATH = "${pkgs.zlib}/lib:${pkgs.openblas}/lib:${pkgs.stdenv.cc.cc.lib}/lib";
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
    version = "2025-08-08";
    mold.enable = pkgs.stdenv.isLinux;
  };
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
    datasets
    gguf[gui]
    huggingface_hub
    matplotlib
    numpy
    onnx
    psutil
    pandas
    scikit-learn
    tabulate
    torch
    torchmetrics
    torchvision
    tqdm
    transformers
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

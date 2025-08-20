{ pkgs, lib, config, inputs, ... }:

{
  cachix.enable = false;

  packages = [
    # General dev.
    pkgs.git pkgs.git-lfs pkgs.openssl pkgs.cmake
    # Rust crates build deps
    pkgs.protobuf pkgs.openssl
    pkgs.llvmPackages.libclang.lib
  ];

  env = {
    OPENSSL_DEV = pkgs.openssl.dev;
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
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
    version = "2025-05-22";
  };
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
    gguf[gui]
    matplotlib
    numpy
    onnx
    psutil
    pandas
    scikit-learn
    tabulate
    torch
    torchvision
    tqdm
    transformers
    '';
  };

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;
  git-hooks.hooks = {
    # actionlint.enable = true;
    check-merge-conflicts.enable = true;
    ripsecrets.enable = true;
    rustfmt = {
      enable = false;
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

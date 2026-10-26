{
  description = "Development shell for knok";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ rust-overlay.overlays.default ];
            };
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        let
          lib = pkgs.lib;
          llvm = pkgs.llvmPackages_23;
          rustLlvm = pkgs.rustc.llvmPackages;
          # Keep the compiler, profiling runtime, and readers from one distribution.
          coverageRust = pkgs.rust-bin.stable.${pkgs.rustc.version}.minimal.override {
            extensions = [ "llvm-tools" ];
          };
          coverageTool = pkgs.cargo-llvm-cov.override {
            rustPlatform = pkgs.makeRustPlatform {
              cargo = coverageRust;
              rustc = coverageRust;
            };
            llvmPackages_22 = llvm;
          };
          compilerVersion = "3.13.0";
          target = pkgs.stdenv.hostPlatform.rust.rustcTarget;
          targetKey = lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] target);
          mlirToolchain = pkgs.symlinkJoin {
            name = "knok-mlir-toolchain";
            paths = [
              llvm.llvm.dev
              llvm.llvm.lib
              llvm.mlir.dev
              llvm.mlir
            ];
          };
          mlirSysLlvmConfig = pkgs.writeShellScriptBin "llvm-config" ''
            case " $* " in
              *" --includedir "*)
                echo "${mlirToolchain}/include"
                ;;
              *" --libdir "*)
                echo "${mlirToolchain}/lib"
                ;;
              *)
                exec "${llvm.llvm.dev}/bin/llvm-config" "$@"
                ;;
            esac
          '';
          python = pkgs.python3.withPackages (ps: [
            ps.pip
            ps.setuptools
            ps.virtualenv
            ps.wheel
          ]);
          darwinPackages = lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.libiconv
          ];
          macosLinker =
            clang:
            pkgs.writeShellScript "knok-macos-linker" ''
              exec ${python}/bin/python ${./scripts/macos-linker.py} --clang ${clang}/bin/clang "$@"
            '';
          shells = {
            default = pkgs.mkShell {
              hardeningDisable = [ "fortify" ];

              packages = [
                pkgs.cargo
                pkgs.rustc
                pkgs.rustfmt
                pkgs.clippy
                pkgs.cmake
                pkgs.ninja
                pkgs.pkg-config
                pkgs.git
                pkgs.cacert
                pkgs.openssl
                pkgs.libxml2
                pkgs.zlib
                llvm.clang
                llvm.libclang
                llvm.llvm
                llvm.mlir
                mlirSysLlvmConfig
                python
              ]
              ++ darwinPackages;

              LIBCLANG_PATH = "${llvm.libclang.lib}/lib";
              MLIR_SYS_230_PREFIX = "${mlirSysLlvmConfig}";
              TABLEGEN_230_PREFIX = "${mlirSysLlvmConfig}";
              "CARGO_TARGET_${targetKey}_LINKER" =
                if pkgs.stdenv.hostPlatform.isDarwin then
                  "${macosLinker llvm.clang}"
                else
                  "${llvm.clang}/bin/clang";
              CC = "${llvm.clang}/bin/clang";
              CXX = "${llvm.clang}/bin/clang++";
              CMAKE_GENERATOR = "Ninja";
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

              shellHook = ''
                export CC="${llvm.clang}/bin/clang"
                export CXX="${llvm.clang}/bin/clang++"

                if [ -z "''${EERIE_PYTHON:-}" ] && [ -z "''${KNOK_IREE_COMPILER:-}" ]; then
                  compiler_venv="''${XDG_CACHE_HOME:-$HOME/.cache}/eerie/iree-compiler-${compilerVersion}-venv"

                  if [ ! -x "$compiler_venv/bin/python" ]; then
                    mkdir -p "$(dirname "$compiler_venv")"
                    ${python}/bin/python -m virtualenv "$compiler_venv" || exit 1
                  fi

                  if ! "$compiler_venv/bin/python" -c 'import importlib.metadata as m; assert m.version("iree-base-compiler") == "${compilerVersion}"' >/dev/null 2>&1; then
                    "$compiler_venv/bin/python" -m pip install --upgrade pip || exit 1
                    "$compiler_venv/bin/pip" install "iree-base-compiler==${compilerVersion}" || exit 1
                  fi

                  export PATH="$compiler_venv/bin:$PATH"
                fi

              '';
            };
          };
        in
        shells
        // {
          coverage = shells.default.overrideAttrs (old: {
            nativeBuildInputs =
              lib.subtractLists [
                pkgs.cargo
                pkgs.rustc
                pkgs.rustfmt
                pkgs.clippy
              ] old.nativeBuildInputs
              ++ [
                coverageRust
                coverageTool
              ];
          });
          cross-lang-lto = shells.default.overrideAttrs (
            old:
            let
              ltoLlvm = rustLlvm;
            in
            {
              nativeBuildInputs = old.nativeBuildInputs ++ [
                ltoLlvm.clang
                ltoLlvm.libclang
                ltoLlvm.llvm
                ltoLlvm.lld
              ];
              shellHook = old.shellHook + ''
                export CC="${ltoLlvm.clang}/bin/clang"
                export CXX="${ltoLlvm.clang}/bin/clang++"
              '';
              LIBCLANG_PATH = "${ltoLlvm.libclang.lib}/lib";
              CARGO_BUILD_TARGET = target;
              "CARGO_TARGET_${targetKey}_LINKER" =
                if pkgs.stdenv.hostPlatform.isDarwin then
                  "${macosLinker ltoLlvm.clang}"
                else
                  "${ltoLlvm.clang}/bin/clang";
              "CARGO_TARGET_${targetKey}_RUSTFLAGS" = "-Clinker-plugin-lto -Clink-arg=-fuse-ld=lld";
            }
          );
        }
      );
    };
}

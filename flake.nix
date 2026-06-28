{
  description = "Development shell for knok";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
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
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        let
          lib = pkgs.lib;
          llvm = pkgs.llvmPackages_22;
          mlirSysLlvmConfig = pkgs.writeShellScriptBin "llvm-config" ''
            case " $* " in
              *" --includedir "*)
                echo "${llvm.mlir.dev}/include"
                ;;
              *" --libdir "*)
                echo "${llvm.mlir}/lib"
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
          darwinPackages = lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
        in
        {
          default = pkgs.mkShell {
            hardeningDisable = [ "fortify" ];

            packages =
              [
                pkgs.cargo
                pkgs.cargo-llvm-cov
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
            MLIR_SYS_220_PREFIX = "${mlirSysLlvmConfig}";
            LLVM_COV = "${llvm.llvm}/bin/llvm-cov";
            LLVM_PROFDATA = "${llvm.llvm}/bin/llvm-profdata";
            CC = "${llvm.clang}/bin/clang";
            CXX = "${llvm.clang}/bin/clang++";
            CMAKE_GENERATOR = "Ninja";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

            shellHook = ''
              export CC="${llvm.clang}/bin/clang"
              export CXX="${llvm.clang}/bin/clang++"

              if [ -z "''${LIB_IREE_COMPILER:-}" ]; then
                : "''${XDG_CACHE_HOME:=$HOME/.cache}"
                export EERIE_IREE_VENV="''${EERIE_IREE_VENV:-$XDG_CACHE_HOME/eerie/iree-compiler-venv}"
                export EERIE_IREE_COMPILER_PACKAGE="''${EERIE_IREE_COMPILER_PACKAGE:-iree-base-compiler==3.11.0}"
                export EERIE_IREE_COMPILER_DISTRIBUTION="''${EERIE_IREE_COMPILER_DISTRIBUTION:-iree-base-compiler}"
                export EERIE_IREE_COMPILER_VERSION="''${EERIE_IREE_COMPILER_VERSION:-3.11.0}"

                if [ ! -x "$EERIE_IREE_VENV/bin/python" ]; then
                  mkdir -p "$(dirname "$EERIE_IREE_VENV")"
                  ${python}/bin/python -m virtualenv "$EERIE_IREE_VENV"
                fi

                if ! "$EERIE_IREE_VENV/bin/python" -c 'import importlib.metadata as m, os; import iree.compiler; assert m.version(os.environ["EERIE_IREE_COMPILER_DISTRIBUTION"]) == os.environ["EERIE_IREE_COMPILER_VERSION"]' >/dev/null 2>&1; then
                  "$EERIE_IREE_VENV/bin/pip" install "$EERIE_IREE_COMPILER_PACKAGE"
                fi

                export PATH="$EERIE_IREE_VENV/bin:$PATH"
                export LIB_IREE_COMPILER="$("$EERIE_IREE_VENV/bin/python" -c 'import iree.compiler as _; print(f"{_.__path__[0]}/_mlir_libs/")')"
              fi

              case "$(uname -s)" in
                Darwin)
                  export RUSTFLAGS="-L native=${llvm.llvm.lib}/lib -L native=${llvm.mlir}/lib -L native=${pkgs.libxml2.out}/lib -L native=${pkgs.zlib.out}/lib -C link-arg=-Wl,-rpath,$LIB_IREE_COMPILER -C link-arg=-Wl,-rpath,${llvm.mlir}/lib ''${RUSTFLAGS:-}"
                  export RUSTDOCFLAGS="-L native=${llvm.llvm.lib}/lib -L native=${llvm.mlir}/lib -L native=${pkgs.libxml2.out}/lib -L native=${pkgs.zlib.out}/lib -C link-arg=-Wl,-rpath,$LIB_IREE_COMPILER -C link-arg=-Wl,-rpath,${llvm.mlir}/lib ''${RUSTDOCFLAGS:-}"
                  export DYLD_FALLBACK_LIBRARY_PATH="${llvm.mlir}/lib:$LIB_IREE_COMPILER''${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
                  ;;
                Linux)
                  export RUSTFLAGS="-L native=${llvm.llvm.lib}/lib -L native=${llvm.mlir}/lib -L native=${pkgs.stdenv.cc.cc.lib}/lib -L native=${pkgs.libxml2.out}/lib -L native=${pkgs.zlib.out}/lib -C link-arg=-Wl,-rpath=$LIB_IREE_COMPILER -C link-arg=-Wl,-rpath=${llvm.mlir}/lib -C link-arg=-Wl,-rpath=${pkgs.stdenv.cc.cc.lib}/lib ''${RUSTFLAGS:-}"
                  export RUSTDOCFLAGS="-L native=${llvm.llvm.lib}/lib -L native=${llvm.mlir}/lib -L native=${pkgs.stdenv.cc.cc.lib}/lib -L native=${pkgs.libxml2.out}/lib -L native=${pkgs.zlib.out}/lib -C link-arg=-Wl,-rpath=$LIB_IREE_COMPILER -C link-arg=-Wl,-rpath=${llvm.mlir}/lib -C link-arg=-Wl,-rpath=${pkgs.stdenv.cc.cc.lib}/lib ''${RUSTDOCFLAGS:-}"
                  export LD_LIBRARY_PATH="${llvm.mlir}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LIB_IREE_COMPILER''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                  ;;
              esac
            '';
          };
        }
      );
    };
}

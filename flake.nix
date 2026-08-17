{
  description = "Entorno de desarrollo reproducible para Tesis (C++ & Python)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "tesis-acj-env";

      buildInputs = with pkgs; [
        cmake
        gcc
        gnumake
        cgal
        boost
        python3
        python3Packages.pip
        python3Packages.virtualenv
		python3Packages.pybind11
        libsForQt5.qtbase
        libGL
        libGLU
        libx11
        fontconfig
        stdenv.cc.cc.lib
        zlib
      ];

      shellHook = ''
        echo "Bienvenido al entorno NixOS de ACJ"
        
        # UNIFICAMOS EL LD_LIBRARY_PATH AQUÍ
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib
            libGL
            libGLU
            libx11
            fontconfig
            zlib
        ])}:$LD_LIBRARY_PATH"
        
        export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.libsForQt5.qtbase.bin}/lib/qt-${pkgs.libsForQt5.qtbase.version}/plugins"

        if [ ! -d ".venv" ]; then
          echo " Creando entorno virtual local..."
          python -m venv .venv
        fi
        
        source .venv/bin/activate
      '';
    };
  };
}

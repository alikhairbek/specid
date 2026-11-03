#!/usr/bin/env bash
# build_portable_win64.sh -- builds the Windows packages of SpecID Studio on Linux (no Windows machine needed):
#   SpecID_Studio_v1.0.1_online.zip   launcher + first-run setup + application + assets (the setup downloads the runtime and packages)
#   SpecID_Studio_v1.0.1_win64.zip    the same with the private CPython 3.12 runtime and the pinned packages already inside
#   SpecID_Studio_assets_v1.0.0.zip   the assets alone (downloaded by the setup of a source checkout)
# Needs: curl, python3 + pip, python3.12 (compileall), x86_64-w64-mingw32-gcc/windres (MinGW-w64), zip.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$HERE/../SpecID_Studio"; W="${WORK:-$PWD/winbuild}"; R="$W/SpecID_Studio"
PBS="https://github.com/astral-sh/python-build-standalone/releases/download/20250529/cpython-3.12.10+20250529-x86_64-pc-windows-msvc-install_only.tar.gz"
mkdir -p "$W/dl" "$W/wheels"; rm -rf "$R"; mkdir -p "$R/app"
# launcher (MinGW cross-compile, with the icon and version resource)
( cd "$W" && cp "$APP/assets/specid.ico" "$HERE/launcher.c" "$HERE/launcher.rc" . && x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher.res \
  && x86_64-w64-mingw32-gcc -municode -mwindows -DUNICODE -D_UNICODE -O2 -s -static launcher.c launcher.res -o "$R/SpecID_Studio.exe" )
# application, setup, helpers
cp "$APP/specid_studio.py" "$APP/engine.py" "$R/app/" && cp -r "$APP/specid" "$APP/assets" "$R/app/" && cp -r "$APP/setup" "$R/setup" && rm -f "$R/setup/setup_log.txt"
cp "$HERE/SpecID_Studio_start.bat" "$HERE/SpecID_Studio_debug.bat" "$HERE/README.txt" "$R/"
find "$R" -name __pycache__ -prune -exec rm -rf {} + ; find "$R" -name "*.log" -delete
( cd "$R/app" && rm -f "$W/SpecID_Studio_assets_v1.0.0.zip" && zip -qr -9 "$W/SpecID_Studio_assets_v1.0.0.zip" assets )
( cd "$W" && rm -f SpecID_Studio_v1.0.1_online.zip && zip -qr -9 SpecID_Studio_v1.0.1_online.zip SpecID_Studio )
# the complete package: runtime + pinned wheels
[ -f "$W/dl/py312.tar.gz" ] || curl -sSL -o "$W/dl/py312.tar.gz" "$PBS"
tar xzf "$W/dl/py312.tar.gz" -C "$W" && rm -rf "$R/runtime" && mv "$W/python" "$R/runtime"
python3 -m pip download --platform win_amd64 --python-version 3.12 --implementation cp --abi cp312 --only-binary=:all: -d "$W/wheels" --quiet -r "$APP/setup/requirements_pinned.txt"
python3 -m pip install --target "$R/runtime/Lib/site-packages" --no-index --find-links "$W/wheels" --platform win_amd64 --python-version 3.12 --implementation cp --abi cp312 --only-binary=:all: --no-compile --quiet -r "$APP/setup/requirements_pinned.txt"
( cd "$R/runtime" && find . -name "*.pdb" -delete && rm -rf include libs Scripts tcl/nmake Lib/idlelib Lib/turtledemo Lib/test Lib/ensurepip DLLs/_test*.pyd \
  && cd Lib/site-packages && rm -rf bin share *.whl && find . -type d -name tests -prune -exec rm -rf {} + && find . -name "*.lib" -delete )
echo ok > "$R/runtime/specid_runtime_ok.txt"
python3.12 -m compileall -q -j 0 --invalidation-mode unchecked-hash "$R/runtime/Lib" "$R/app/engine.py" "$R/app/specid" || true
( cd "$W" && rm -f SpecID_Studio_v1.0.1_win64.zip && zip -qr -9 SpecID_Studio_v1.0.1_win64.zip SpecID_Studio ) && ls -la "$W"/SpecID_Studio_*.zip

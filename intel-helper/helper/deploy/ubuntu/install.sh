#!/usr/bin/env bash
# Install omlx-intel-helper on Ubuntu (tested nowhere yet: written ahead of the
# NUC 15 Pro move; read it before running, it uses sudo for the GPU runtime).
#
#   ./install.sh [--no-service]
#
# What it does:
#   1. checks the Intel GPU is visible (/dev/dri/renderD*) and you are in the
#      render group
#   2. installs Intel's Level Zero / compute runtime from Ubuntu's archive
#   3. makes a venv in ~/omlx-intel-helper-venv with torch (XPU) and the helper
#   4. checks torch.xpu sees the GPU
#   5. installs a systemd *user* service (unless --no-service)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
VENV="$HOME/omlx-intel-helper-venv"
SERVICE=1
[ "${1:-}" = "--no-service" ] && SERVICE=0

echo "== Intel GPU"
if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
  echo "No /dev/dri/renderD*: the Intel GPU driver is not active. Arc 140T (Arrow Lake-H)"
  echo "needs a recent kernel (6.11 or newer: Ubuntu HWE or 25.04+). Check: uname -r"
  exit 1
fi
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "Adding $USER to the render and video groups (log out and in afterwards)"
  sudo usermod -aG render,video "$USER"
fi

echo "== Level Zero and the compute runtime (Ubuntu archive)"
sudo apt-get update
# Names as in Ubuntu 25.04+/26.04; on 24.04 add Intel's graphics PPA first
# (see https://dgpu-docs.intel.com/driver/client/overview.html).
sudo apt-get install -y python3-venv python3-pip clinfo \
  libze1 libze-intel-gpu1 intel-opencl-icd

echo "== Python environment: $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/xpu
"$VENV/bin/pip" install "$SRC"

echo "== torch.xpu"
"$VENV/bin/python" - <<'PY'
import torch
assert hasattr(torch, "xpu") and torch.xpu.is_available(), "torch.xpu does not see the GPU"
p = torch.xpu.get_device_properties(0)
print("XPU:", p.name, f"{p.total_memory / 2**30:.1f} GiB visible, integrated={getattr(p, 'is_integrated_gpu', '?')}")
PY

if [ "$SERVICE" = 1 ]; then
  echo "== systemd user service"
  mkdir -p "$HOME/.config/systemd/user"
  sed "s|@VENV@|$VENV|g" "$HERE/omlx-intel-helper.service" > "$HOME/.config/systemd/user/omlx-intel-helper.service"
  systemctl --user daemon-reload
  systemctl --user enable --now omlx-intel-helper.service
  loginctl enable-linger "$USER" || true
  echo "Pairing code: journalctl --user -u omlx-intel-helper -f   (look for PAIRING CODE)"
fi
echo "Done. Allow ports 8850-8852/tcp from the Mac if a firewall is on."

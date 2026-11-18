# SPDX-License-Identifier: Apache-2.0
"""What this machine has: Intel GPUs, memory, disk and network addresses.

Works on Windows and Linux. GPUs come from ``torch.xpu``; an integrated Arc
(Core Ultra) has no memory of its own, so its budget is what the driver shows
capped by the free system RAM minus a share left to the OS.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
from pathlib import Path
from typing import Any

import psutil

# RAM left to the OS and other apps while an integrated GPU holds layers.
_OS_BYTES = int(float(os.environ.get("OMLX_HELPER_OS_RESERVE_GIB", "6")) * (1 << 30))


def _torch():
    try:
        import torch  # noqa: PLC0415 - heavy, loaded on first use only
    except ImportError:
        return None
    return torch if hasattr(torch, "xpu") and torch.xpu.is_available() else None


def gpus(only: list[int] | None = None) -> list[dict[str, Any]]:
    """Intel GPUs as {index, name, total, free, integrated}, sizes in bytes."""
    torch = _torch()
    if torch is None:
        return []
    found = []
    ram_total, ram_available = memory()
    for index in range(torch.xpu.device_count()):
        if only is not None and index not in only:
            continue
        props = torch.xpu.get_device_properties(index)
        total = int(props.total_memory)
        integrated = bool(getattr(props, "is_integrated_gpu", False))
        try:
            free_gpu, _ = torch.xpu.mem_get_info(index)
        except Exception:  # noqa: BLE001 - older builds have no mem_get_info
            free_gpu = total
        free = int(free_gpu)
        if integrated:
            free = min(free, max(0, ram_available - _OS_BYTES))
        found.append(
            {
                "index": index,
                "name": props.name,
                "total": total,
                "free": free,
                "integrated": integrated,
            }
        )
    return found


def memory_limit(gpu: dict[str, Any]) -> int:
    """Bytes a stage on *gpu* may hold before it hands a prompt back.

    On an integrated GPU the OS keeps its share of the RAM.
    """
    if gpu.get("integrated"):
        return max(0, min(gpu["total"], gpu["free"] + _OS_BYTES) - _OS_BYTES)
    return int(gpu["total"] * 0.95)


def wsl() -> bool:
    """Whether this runs on WSL."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def chip_label(found: list[dict[str, Any]]) -> str:
    """ "Intel Arc 140T" or "2x Intel Arc B60" for the device card on the host."""
    names = [g["name"] for g in found]
    if not names:
        return "No Intel GPU"
    if len(set(names)) == 1:
        return names[0] if len(names) == 1 else f"{len(names)}x {names[0]}"
    return " + ".join(names)


def memory() -> tuple[int, int]:
    """(total, available) system RAM in bytes."""
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def disk_free(path: Path) -> int:
    while not path.is_dir():
        path = path.parent
    return shutil.disk_usage(path).free


# Adapters of containers, VMs and VPNs, never a link to a Mac.
_VIRTUAL = (
    "docker", "br-", "veth", "virbr", "vethernet", "vmware", "virtualbox",
    "hyper-v", "wsl", "loopback", "tailscale", "zerotier", "tun", "tap",
)
_WIRELESS = ("wi-fi", "wifi", "wlan", "wireless", "wlp")


def _real(name: str) -> bool:
    low = name.lower()
    return not any(low.startswith(v) or v in low for v in _VIRTUAL)


def _wireless(name: str) -> bool:
    low = name.lower()
    return any(w in low for w in _WIRELESS)


def local_ipv4() -> list[str]:
    """Routable IPv4 addresses of this machine, the default route's first."""
    found = []
    for name, addrs in psutil.net_if_addrs().items():
        if not _real(name):
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith(
                ("127.", "169.254.")
            ):
                found.append(addr.address)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # No packet is sent; this only asks the OS for the route.
            probe.connect(("192.0.2.1", 9))
            primary = probe.getsockname()[0]
    except OSError:
        primary = ""
    if primary and primary in found:
        found.remove(primary)
    return ([primary] if primary and not primary.startswith("127.") else []) + found


def links() -> list[dict[str, Any]]:
    """Wired interfaces that are up: IPv4 networks, IPv6 link-local, Mbit/s.

    A direct cable to the Mac gets no DHCP lease; its IPv6 link-local address
    is the one the Mac can reach.
    """
    stats = psutil.net_if_stats()
    found = []
    for name, addrs in psutil.net_if_addrs().items():
        st = stats.get(name)
        if st is None or not st.isup or not _real(name) or _wireless(name):
            continue
        ipv4, link_local = [], None
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.netmask:
                if addr.address.startswith(("127.", "169.254.")):
                    continue
                prefix = ipaddress.ip_network(
                    f"{addr.address}/{addr.netmask}", strict=False
                ).prefixlen
                ipv4.append(f"{addr.address}/{prefix}")
            elif addr.family == socket.AF_INET6 and addr.address.lower().startswith(
                "fe80"
            ):
                link_local = addr.address.split("%", 1)[0]
        if ipv4 or link_local:
            found.append(
                {
                    "interface": name,
                    "ipv4": ipv4,
                    "link_local": link_local,
                    "mbps": st.speed if st.speed > 0 else None,
                }
            )
    return found


def link_mbps(address: str) -> int | None:
    """Negotiated speed of the wired interface that holds *address*. Wi-Fi
    interfaces are not listed, so a helper on Wi-Fi reports None."""
    address = address.split("%", 1)[0]
    for link in links():
        held = [cidr.split("/", 1)[0] for cidr in link["ipv4"]]
        if address in held or address == link["link_local"]:
            return link["mbps"]
    return None

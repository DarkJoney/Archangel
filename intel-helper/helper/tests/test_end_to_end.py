# SPDX-License-Identifier: Apache-2.0
"""A real worker process serves the Mac-recorded contract fixtures.

Everything but the Mac's GPU is real: oMLX's own host code starts the session
over the control channel (``agent.launch``), the supervisor starts a stage
worker process with the real family stage, and the fixture's chunks go over
the data port with the host's inputs. What comes back, byte for byte the
arrays of the wire, must equal what the Mac recorded.

Needs the oMLX source importable (Linux / WSL: ``PYTHONPATH=<omlx tree>``).
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omlx_intel_helper import families, gpus
from omlx_intel_helper.control import ControlServer
from omlx_intel_helper.files import ModelStore
from omlx_intel_helper.state import State
from omlx_intel_helper.supervisor import Supervisor
from omlx_intel_helper.worker import Upstream

FIXTURES = Path(__file__).parent / "fixtures/prefill_contract"


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


def _check(got, want, rtol, atol, label):
    assert sorted(got) == sorted(want), label
    for name, (data, dtype) in got.items():
        expected, expected_dtype = want[name]
        assert (dtype, data.dtype, data.shape) == (
            expected_dtype,
            expected.dtype,
            expected.shape,
        ), label + name
        if data.dtype.kind == "f":
            np.testing.assert_allclose(data, expected, rtol=rtol, atol=atol, err_msg=label + name)
        else:
            np.testing.assert_array_equal(data, expected, err_msg=label + name)


@pytest.mark.parametrize("family", ["qwen3_5", "qwen4_exp"])
def test_a_real_worker_serves_the_contract_fixture(monkeypatch, family):
    pytest.importorskip("omlx")
    from omlx.prefill_helper import agent

    fixture = FIXTURES / family
    meta = json.loads((fixture / "meta.json").read_text())
    data = np.load(fixture / "case.npz")
    arrays = {n: (data[n], d) for n, d, _, _ in meta["manifest"]}
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    chunks = sizes[skip:]
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]

    root = Path(tempfile.mkdtemp(prefix="och-"))
    key = bytes(range(32))
    state = State(root / "state", name="intel1")
    state.save_host("studio", {"cluster_key": key.hex(), "name": "Studio"})
    gpu = {"index": 0, "name": "Intel(R) Arc(TM) test", "total": 48 << 30,
           "free": 48 << 30, "integrated": False}
    monkeypatch.setattr(gpus, "gpus", lambda only=None: [gpu])
    monkeypatch.setattr(gpus, "links", lambda: [])
    shutil.copytree(fixture / "model", root / "models" / "Model")

    supervisor = Supervisor(None, root / "run")
    control = ControlServer(
        state, ModelStore(root / "models"), supervisor, families.contracts(), bind=""
    )
    control.start()
    me = SimpleNamespace(node_id="studio")
    monkeypatch.setattr(agent, "get_node_identity", lambda: me)
    # The family's real Mac adapter decides which contract it asks for, when it
    # imports here (it needs MLX and mlx-vlm); otherwise the helper's own number.
    try:
        adapter = agent.adapter_for(family)
        assert adapter is None or adapter.contract == families.FAMILIES[family].contract
    except ImportError:
        monkeypatch.setattr(
            agent, "adapter_for",
            lambda t: SimpleNamespace(contract=families.FAMILIES[family].contract),
        )
    target = agent.AgentTarget(
        state.node_id, "intel1", "127.0.0.1", "ethernet", control.port, key
    )
    process = None
    try:
        info = agent.remote_info(target)
        assert info["contracts"] == families.contracts()
        assert agent.incompatible(info, adapter_for_check(agent, family), "Intel1") is None
        process = agent.launch(
            target, str(root / "models" / "Model"), family, k,
            host_name="Studio", warm_tokens=8,
        )
        assert process.hello["layers"] == k and process.hello["model_type"] == family
        host = Upstream(
            socket.create_connection(("127.0.0.1", process.port), timeout=30), process.token
        )
        prefix = {n: v for n, v in arrays.items() if n.startswith("p:")}
        host.submit(
            {"op": "job", "job_id": "j", "chunks": chunks, "layers": k, "feed": True},
            {"tokens": (tokens.reshape(1, -1), "int32"), **prefix},
        )
        for index in range(len(chunks)):
            host.send(
                {"op": "inputs", "job_id": "j", "index": index},
                _group(arrays, f"in{index}:"),
            )
            reply, got = host.take("j", index)
            assert reply["op"] == "chunk" and reply["size"] == chunks[index]
            boundary = {n[2:]: v for n, v in got.items() if n.startswith("b:")}
            own = {n[2:]: v for n, v in got.items() if n.startswith("s:")}
            _check(boundary, _group(arrays, f"b{index}:"), meta["rtol"], meta["atol"], f"b{index}:")
            _check(own, _group(arrays, f"s{index}:"), meta["rtol"], meta["atol"], f"s{index}:")
        host.cancel("j")
        process.stop()
        process = None
        assert supervisor.status() is None
    finally:
        if process is not None:
            process.stop()
        supervisor.stop()
        control.stop()
        shutil.rmtree(root, ignore_errors=True)


def adapter_for_check(agent, family):
    """The adapter object ``incompatible`` compares contracts with."""
    try:
        return agent.adapter_for(family)
    except ImportError:
        return SimpleNamespace(model_types=(family,), contract=families.FAMILIES[family].contract)

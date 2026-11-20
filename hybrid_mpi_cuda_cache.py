"""Four-rank heat diffusion and a shared-window counter benchmark."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

import numpy as np
from mpi4py import MPI
from numba import cuda, float64, njit, types
from numba.core import cgutils
from numba.extending import intrinsic

N = 256
P = N + 2
TILE = 16
STEPS = 300
ITERATIONS = 40_000_000
DX = 1 / 513
DAMP = 0.15
OPPOSITE = (1, 0, 3, 2)


def init_hybrid_context(user):
    world = MPI.COMM_WORLD
    if world.size != 4:
        raise ValueError("Start exactly four MPI ranks.")
    cart = world.Create_cart((2, 2), periods=(False, False), reorder=False)
    node = world.Split_type(MPI.COMM_TYPE_SHARED, key=world.rank)
    if node.size != 4:
        raise RuntimeError("All four ranks must be on the same node.")
    cuda.select_device(0)
    if world.rank == 0:
        name, uuid = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=name,uuid", "--format=csv,noheader"],
            text=True).strip().split(", ")
        cpu = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
        model = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)
        signature = platform.machine() + "|" + (model[1] if model else "") + "|" + cpu.strip()
        line = int(Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size").read_text())
        seed = int.from_bytes(hashlib.sha256(f"{user}::{signature}::{uuid}".encode()).digest()[:8], "big")
        info = dict(github_username=user, student_id="240103179", student_name="Bakytkerey Abulsagit",
                    gpu_name=name, gpu_uuid=uuid, cpu_signature=signature, cpu_line_bytes=line,
                    INSTANCE_SEED=seed)
    else:
        info = None
    info = world.bcast(info, root=0)
    if "T4" not in info["gpu_name"] or info["cpu_line_bytes"] != 64:
        raise RuntimeError("Select a T4 runtime with 64-byte host cache lines.")
    win = MPI.Win.Allocate_shared(64 * 8 + 127 if node.rank == 0 else 0, 1, comm=node)
    mem, _ = win.Shared_query(0)
    raw = np.frombuffer(mem, dtype=np.uint8)
    offset = node.bcast((-raw.ctypes.data) % 128 if node.rank == 0 else None, root=0)
    view = np.frombuffer(mem, dtype=np.uint64, count=64, offset=offset)
    if node.allreduce(view.ctypes.data % 128, op=MPI.MAX):
        raise RuntimeError("Inconsistent shared-window alignment.")
    win.Lock_all()
    neighbors = cart.Shift(0, 1) + cart.Shift(1, 1)
    info["rank_cpu_affinity"] = world.allgather(sorted(os.sched_getaffinity(0)))
    return cart, node, win, view, neighbors, info


def initialize_fields(coords, seed):
    y, x = np.ogrid[:P, :P]
    x = (coords[1] * N + x) * DX
    y = (coords[0] * N + y) * DX
    phase = (seed % 65536) * (2 * np.pi / 65536)
    wave = np.cos(2 * np.pi * x + phase) * np.cos(2 * np.pi * y)
    u = 1 + 0.3 * wave + 0.1 * np.cos(4 * np.pi * y)
    k = 0.08 + 0.02 * np.sin(2 * np.pi * x + phase) * np.cos(2 * np.pi * y)
    source = 0.02 * wave
    return u, k, source


@cuda.jit
def pack_halos_kernel(u, buf):
    i = cuda.grid(1)
    if i < N:
        buf[0, i] = u[1, i+1]
        buf[1, i] = u[N, i+1]
        buf[2, i] = u[i+1, 1]
        buf[3, i] = u[i+1, N]


@cuda.jit
def unpack_halos_kernel(u, buf, top, bottom, left, right):
    i = cuda.grid(1)
    if i < N:
        j = i + 1
        # Mirror the nearest interior value on insulated physical faces.
        u[0, j] = buf[0, i] if top else u[1, j]
        u[N+1, j] = buf[1, i] if bottom else u[N, j]
        u[j, 0] = buf[2, i] if left else u[j, 1]
        u[j, N+1] = buf[3, i] if right else u[j, N]


@cuda.jit
def tiled_stencil_kernel(u, out, k, source, dt, region):
    tile = cuda.shared.array((18, 18), float64)
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    if region == 1 and 0 < bx < N//TILE-1 and 0 < by < N//TILE-1:
        return
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x, y = bx*TILE + tx+1, by*TILE + ty+1
    lx, ly = tx+1, ty+1
    tile[ly, lx] = u[y, x]
    # The interior stream never reads ghost cells being filled by the other stream.
    lo, hi = 0, N+1
    if region == 0:
        lo, hi = 1, N
    if tx == 0:
        tile[ly, 0] = u[y, max(lo, x-1)]
    if tx == TILE-1:
        tile[ly, TILE+1] = u[y, min(hi, x+1)]
    if ty == 0:
        tile[0, lx] = u[max(lo, y-1), x]
    if ty == TILE-1:
        tile[TILE+1, lx] = u[min(hi, y+1), x]
    cuda.syncthreads()
    border = x == 1 or x == N or y == 1 or y == N
    if (region == 1) == border:
        c, kc = tile[ly, lx], k[y, x]
        flux = ((kc+k[y, x-1])*(tile[ly, lx-1]-c) +
                (kc+k[y, x+1])*(tile[ly, lx+1]-c) +
                (kc+k[y-1, x])*(tile[ly-1, lx]-c) +
                (kc+k[y+1, x])*(tile[ly+1, lx]-c))
        out[y, x] = c + dt * (0.5*flux/(DX*DX) + source[y, x] - DAMP*c)


class HaloExchange:
    def __init__(self, cart, neighbors):
        self.cart = cart
        self.links = [(d, peer) for d, peer in enumerate(neighbors) if peer != MPI.PROC_NULL]
        self.present = tuple(peer != MPI.PROC_NULL for peer in neighbors)
        self.boundary, self.interior = cuda.stream(), cuda.stream()
        self.send = cuda.pinned_array((4, N), np.float64)
        self.recv = cuda.pinned_array((4, N), np.float64)
        self.recv.fill(0)
        self.ds = cuda.device_array_like(self.send)
        self.dr = cuda.device_array_like(self.recv)

    def step_overlapped(self, u, out, k, src, dt, overlap):
        b, s = self.boundary, self.interior
        grid, block = (N//TILE, N//TILE), (TILE, TILE)
        requests = [self.cart.Irecv(self.recv[d], peer, OPPOSITE[d]) for d, peer in self.links]
        pack_halos_kernel[2, 128, b](u, self.ds)
        for d, _ in self.links:
            self.ds[d].copy_to_host(self.send[d], stream=b)
        if overlap:
            tiled_stencil_kernel[grid, block, s](u, out, k, src, dt, 0)
        b.synchronize()
        for d, peer in self.links:
            requests.append(self.cart.Isend(self.send[d], peer, d))
        MPI.Request.Waitall(requests)
        for d, _ in self.links:
            self.dr[d].copy_to_device(self.recv[d], stream=b)
        unpack_halos_kernel[2, 128, b](u, self.dr, *self.present)
        if not overlap:
            tiled_stencil_kernel[grid, block, b](u, out, k, src, dt, 0)
        tiled_stencil_kernel[grid, block, b](u, out, k, src, dt, 1)
        b.synchronize()
        if overlap:
            s.synchronize()


@intrinsic
def increment_slot(typingctx, array, index):
    sig = types.void(array, index)
    def codegen(context, builder, signature, args):
        ary = context.make_array(signature.args[0])(context, builder, args[0])
        ptr = builder.gep(ary.data, [args[1]])
        one = context.get_constant(types.uint64, 1)
        builder.atomic_rmw("add", ptr, one, "monotonic")
        return context.get_dummy_value()
    return sig, codegen


@njit(nogil=True)
def host_shm_accumulate(view, rank, stride, count):
    slot = rank * stride
    for _ in range(count):
        increment_slot(view, slot)


def benchmark_shm_strides(node, win, view):
    rank = node.rank
    host_shm_accumulate(view, rank, 16, 0)
    samples = {s: [] for s in (1, 8, 16)}
    for order in ((1, 8, 16), (8, 16, 1), (16, 1, 8)):
        for stride in order:
            if rank == 0:
                view.fill(0)
            win.Sync()
            node.Barrier()
            win.Sync()
            node.Barrier()
            start = MPI.Wtime()
            host_shm_accumulate(view, rank, stride, ITERATIONS)
            elapsed = MPI.Wtime() - start
            samples[stride].append(node.allreduce(elapsed, op=MPI.MAX)*1000)
            win.Sync()
            node.Barrier()
            win.Sync()
            if not np.all(view[np.arange(node.size)*stride] == ITERATIONS):
                raise ArithmeticError("Unexpected shared counter total")
            node.Barrier()
    ir = host_shm_accumulate.inspect_llvm(host_shm_accumulate.signatures[0])
    instructions = [line.strip() for line in ir.splitlines() if "atomicrmw add" in line and "monotonic" in line]
    if not instructions:
        raise RuntimeError("No in-place counter update in optimized LLVM")
    times = {str(s): float(np.median(v)) for s, v in samples.items()}
    return times, samples, instructions


def measure_field(cart, arr):
    a = arr.copy_to_host()[1:-1, 1:-1]
    return (cart.allreduce(float(np.sum(a))*DX**2),
            cart.allreduce(float(np.sum(a*a))*0.5*DX**2))


def solve(cart, halo, initial, k, src, dt, overlap, source_total):
    u, out = cuda.to_device(initial), cuda.to_device(initial)
    # Compile both kernels and complete one exchange before starting the clock.
    halo.step_overlapped(u, out, k, src, dt, overlap)
    u.copy_to_device(initial)
    mass0, energy0 = measure_field(cart, u)
    energy = {"0": energy0}
    mass = {"0": mass0}
    balance = 0.0
    cart.Barrier()
    elapsed = 0.0
    for start_step in (0, 100, 200):
        cart.Barrier()
        start = MPI.Wtime()
        for _ in range(100):
            halo.step_overlapped(u, out, k, src, dt, overlap)
            u, out = out, u
        elapsed += cart.allreduce(MPI.Wtime()-start, op=MPI.MAX)
        step = start_step + 100
        m, e = measure_field(cart, u)
        mass[str(step)], energy[str(step)] = m, e
        decay = math.exp(step * math.log1p(-dt * DAMP))
        expected = decay * mass0 + (1-decay) * source_total / DAMP
        balance = max(balance, abs(m - expected))
        if not np.isfinite([m, e]).all() or balance >= 1e-12:
            raise ArithmeticError(f"Step {step}: thermal balance error {balance}")
    if any(b > a + 1e-12 for a, b in zip(list(energy.values()), list(energy.values())[1:])):
        raise ArithmeticError("Squared-field energy grew for this test field")
    return u.copy_to_host()[1:-1, 1:-1], elapsed*1000, energy, mass, balance


def audit_ptx():
    ptx = next(iter(tiled_stencil_kernel.inspect_asm().values()))
    ops = re.findall(r"\b(?:(?:ld|st)\.shared(?:\.[a-z0-9]+)+|bar\.sync)\b", ptx)
    counts = {op: ops.count(op) for op in sorted(set(ops) | {"st.shared.f64", "ld.shared.f64", "bar.sync"})}
    if not all(any(op.startswith(prefix) and n for op, n in counts.items())
               for prefix in ("st.shared.", "ld.shared.", "bar.sync")):
        raise RuntimeError(f"Incomplete tile PTX: {counts}")
    return counts


def main(user):
    cart, node, win, view, neighbors, info = init_hybrid_context(user)
    u, k, src = initialize_fields(cart.Get_coords(cart.rank), info["INSTANCE_SEED"])
    kmax = cart.allreduce(float(k[1:-1, 1:-1].max()), op=MPI.MAX)
    limit = DX**2/(4*kmax)
    dt = 0.8 * limit
    source_total = cart.allreduce(float(src[1:-1, 1:-1].sum()) * DX**2)
    halo = HaloExchange(cart, neighbors)
    kd, sd = cuda.to_device(k), cuda.to_device(src)
    serial, ts, es, ms, bs = solve(cart, halo, u, kd, sd, dt, False, source_total)
    parallel, tp, ep, mp, bp = solve(cart, halo, u, kd, sd, dt, True, source_total)
    diff = cart.allreduce(float(np.max(np.abs(serial-parallel))), op=MPI.MAX)
    if not diff < 1e-12:
        raise ArithmeticError(f"Parity check failed: {diff}")
    ptx = audit_ptx()
    times, samples, ir = benchmark_shm_strides(node, win, view)
    win.Unlock_all()
    win.Free()
    if cart.rank == 0:
        info.update(ranks=4, grid=[512,512], steps=STEPS, iterations_per_rank=ITERATIONS,
                    kappa_max=kmax, dx=DX, dt=dt, dt_max=limit,
                    gpu_serialized_ms=ts, gpu_overlapped_ms=tp, max_diff=diff,
                    global_energy=ep, serialized_energy=es, global_heat=mp, serialized_heat=ms,
                    source_integral=source_total, max_heat_balance_error=max(bs,bp),
                    ptx_counts=ptx, llvm_atomicrmw_count=len(ir), llvm_instructions=ir,
                    shm_stride_ms=times, shm_samples_ms=samples,
                    false_sharing_slowdown=times["1"]/times["8"])
        print(json.dumps(info, indent=2), flush=True)
        target = Path("hybrid_lab_manifest.json")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2) + "\n")
        tmp.replace(target)
    node.Free()
    cart.Free()


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)

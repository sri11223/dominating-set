#!/usr/bin/env python3
"""Run a candidate solver with network use blocked at the syscall level.

A seccomp filter is installed that makes the socket(2) syscall fail with EPERM,
then the candidate is executed in-process. The block is deterministic (always
applied), and a solver that never touches the network is unaffected, so a
compliant solver can never fail because of this control. A solver that attempts
network I/O gets an OSError and cannot complete.

Usage: _netblock.py CANDIDATE INPUT_JSON OUTPUT_DIRECTORY
"""
import ctypes
import os
import runpy
import sys

_AUDIT_ARCH_X86_64 = 0xC000003E
_SYS_SOCKET = 41
_SYS_SECCOMP = 317
_SECCOMP_SET_MODE_FILTER = 1
_PR_SET_NO_NEW_PRIVS = 38
_EPERM = 1
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
_BPF_LD, _BPF_W, _BPF_ABS = 0x00, 0x00, 0x20
_BPF_JMP, _BPF_JEQ, _BPF_K = 0x05, 0x10, 0x00
_BPF_RET = 0x06


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def install_socket_block() -> bool:
    """Install a seccomp filter that fails the socket syscall with EPERM.

    Returns True when the filter is active. Every other syscall is allowed, so
    ordinary computation (including numpy/scipy/sympy) is unaffected.
    """
    program = [
        # load arch; if not x86_64, skip to allow (the container is x86_64).
        (_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, 4),
        (_BPF_JMP | _BPF_JEQ | _BPF_K, 0, 3, _AUDIT_ARCH_X86_64),
        # load syscall number; if socket, block, else allow.
        (_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, 0),
        (_BPF_JMP | _BPF_JEQ | _BPF_K, 0, 1, _SYS_SOCKET),
        (_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_ERRNO | _EPERM),
        (_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_ALLOW),
    ]
    filters = (_SockFilter * len(program))(*(_SockFilter(*entry) for entry in program))
    prog = _SockFprog(len(program), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    # no_new_privs is required before a filter can be installed unprivileged.
    libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    return libc.syscall(_SYS_SECCOMP, _SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(prog)) == 0


def main() -> None:
    blocked = install_socket_block()
    if os.environ.get("NETBLOCK_PROBE") == "1":
        # Canary mode: report whether socket creation is blocked, then exit.
        if not blocked:
            sys.stdout.write("unfiltered")
            return
        try:
            import socket
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sys.stdout.write("open")
        except OSError:
            sys.stdout.write("blocked")
        return
    target = sys.argv[1]
    if not blocked:
        # The filter could not be installed; refuse to run without the control.
        raise SystemExit("network block could not be installed")
    sys.argv = [target] + sys.argv[2:]
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        raise SystemExit(code)


if __name__ == "__main__":
    main()

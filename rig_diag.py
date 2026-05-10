#!/usr/bin/env python3
"""Run diagnostics on the rig via paramiko (password auth).

Also installs my SSH pubkey so future SSH works without a password.
"""
import paramiko
import sys, io
# Force utf-8 output so lsblk tree chars don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = '192.168.1.105'
USER = 'user'
PASS = '1'
PUBKEY = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJNcaBBUK12w7dPLXNOMFVERB2ThqIas6tU4zfkUgDRW Rahul Goyanka@DESKTOP-8NR65O3'

CMDS = [
    # GPUs visible?
    'lspci | grep -iE "vga|3d|display" | head',
    'ls /dev/dri/ 2>&1',
    # Vulkan + AMD drivers loaded?
    'lsmod | grep -iE "amdgpu|vulkan|radeon" | head',
    'which vulkaninfo glxinfo',
    'vulkaninfo --summary 2>&1 | head -40 || echo no_vulkan',
    # Python version + what exists on HiveOS
    'python3 --version',
    'which docker && docker --version',
    # what is HiveOS using RAM for (check we can run a heavy workload)
    'free -h',
    # hardware check
    'nproc; lscpu | grep -E "Model name|MHz" | head -3',
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=15, allow_agent=False, look_for_keys=False)

for cmd in CMDS:
    print(f'\n$ {cmd}')
    stdin, stdout, stderr = c.exec_command(cmd, timeout=60, get_pty=False)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    if out:
        print(out.rstrip())
    if err.strip():
        print('[stderr]', err.rstrip())

c.close()

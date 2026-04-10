"""
AI Rig Client — Wake, health check, and shutdown for remote agents.

Drop this file into any project that needs GPU inference from the rig.
Call ensure_rig_ready() before sending work. Call rig_sleep() when done.

Usage:
    from rig_client import ensure_rig_ready, rig_sleep, RIG_PORTS, RIG_BASE_URL

    # Before processing
    ready = ensure_rig_ready()    # wakes rig if sleeping, waits for GPUs
    if not ready:
        print("Rig unavailable")
        exit(1)

    # Send work to any GPU
    # POST http://192.168.1.107:8001/v1/chat/completions
    # POST http://192.168.1.107:8002/v1/chat/completions
    # ... through 8007

    # After processing (optional — rig auto-sleeps after 30 min idle)
    rig_sleep()
"""

import socket
import struct
import time
import logging

logger = logging.getLogger(__name__)

# ── Rig Configuration ─────────────────────────────────────────────────────
RIG_IP = "192.168.1.107"
RIG_MAC = "a8:a1:59:91:d2:96"
RIG_BROADCAST = "192.168.1.255"
RIG_PORTS = [8080, 8081, 8082, 8083, 8084, 8085, 8086]
RIG_LB_PORT = 4000  # nginx load balancer — ONE endpoint for all 7 GPUs
RIG_BASE_URL = f"http://{RIG_IP}:{RIG_LB_PORT}"  # use this for all requests
RIG_API_URL = f"{RIG_BASE_URL}/v1/chat/completions"  # direct chat endpoint

# Timeouts
WOL_WAIT_SSH = 120        # max seconds to wait for SSH after WoL
WOL_WAIT_GPU = 180        # max seconds to wait for GPUs after SSH is up
HEALTH_TIMEOUT = 5        # seconds per port health check


def _send_wol():
    """Send Wake-on-LAN magic packet."""
    mac_bytes = bytes.fromhex(RIG_MAC.replace(":", ""))
    magic = b'\xff' * 6 + mac_bytes * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(magic, (RIG_BROADCAST, 9))


def _check_port(port: int, timeout: float = HEALTH_TIMEOUT) -> bool:
    """Check if a TCP port is responding."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((RIG_IP, port))
        s.close()
        return True
    except Exception:
        return False


def is_rig_awake() -> bool:
    """Check if rig is awake (SSH responding)."""
    return _check_port(22, timeout=3)


def get_gpu_status() -> dict:
    """Check which GPU ports are responding.

    Returns:
        dict with 'ready' (list of ports), 'down' (list of ports), 'total', 'ready_count'
    """
    ready = []
    down = []
    for port in RIG_PORTS:
        if _check_port(port):
            ready.append(port)
        else:
            down.append(port)
    return {
        "ready": ready,
        "down": down,
        "total": len(RIG_PORTS),
        "ready_count": len(ready),
    }


def wake_rig(wait_for_ssh: bool = True) -> bool:
    """Send WoL packet to wake the rig.

    Args:
        wait_for_ssh: if True, block until SSH is responding

    Returns:
        True if rig is awake
    """
    if is_rig_awake():
        logger.info("Rig already awake")
        return True

    logger.info("Sending Wake-on-LAN packet to %s", RIG_MAC)
    _send_wol()

    if not wait_for_ssh:
        return True

    logger.info("Waiting for rig to boot (up to %ds)...", WOL_WAIT_SSH)
    for i in range(WOL_WAIT_SSH // 5):
        time.sleep(5)
        if is_rig_awake():
            logger.info("Rig awake after %ds", (i + 1) * 5)
            return True

    logger.error("Rig did not wake within %ds", WOL_WAIT_SSH)
    return False


def wait_for_gpus(min_gpus: int = 1) -> dict:
    """Wait until at least min_gpus are ready.

    Args:
        min_gpus: minimum number of GPUs required (default 1, max 7)

    Returns:
        gpu status dict, or None if timeout
    """
    logger.info("Waiting for at least %d GPU(s) (up to %ds)...", min_gpus, WOL_WAIT_GPU)
    for i in range(WOL_WAIT_GPU // 10):
        time.sleep(10)
        status = get_gpu_status()
        logger.info("  %d/%d GPUs ready", status["ready_count"], status["total"])
        if status["ready_count"] >= min_gpus:
            return status

    logger.error("Not enough GPUs ready within %ds", WOL_WAIT_GPU)
    return None


def ensure_rig_ready(min_gpus: int = 6) -> dict | None:
    """Wake the rig and wait until GPUs are ready. One call does everything.

    Args:
        min_gpus: minimum GPUs required before returning (default 6 of 7)

    Returns:
        gpu status dict with 'ready' ports list, or None if failed

    Example:
        status = ensure_rig_ready()
        if status:
            print(f"{status['ready_count']} GPUs ready on ports {status['ready']}")
            # Send work to any port in status['ready']
        else:
            print("Rig unavailable")
    """
    # Step 1: Check if already ready
    if is_rig_awake():
        status = get_gpu_status()
        if status["ready_count"] >= min_gpus:
            logger.info("Rig ready: %d/%d GPUs", status["ready_count"], status["total"])
            return status
        logger.info("Rig awake but only %d/%d GPUs ready, waiting...", status["ready_count"], status["total"])
        result = wait_for_gpus(min_gpus)
        if result:
            return result

    # Step 2: Wake the rig
    if not wake_rig(wait_for_ssh=True):
        return None

    # Step 3: Wait for GPUs
    return wait_for_gpus(min_gpus)


def rig_sleep():
    """Suspend rig to RAM. Models stay in VRAM, wakes in ~20s via WoL."""
    if not is_rig_awake():
        logger.info("Rig already sleeping/off")
        return

    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(RIG_IP, username='user', password='1', timeout=5)
        ssh.exec_command('echo 1 | sudo -S systemctl suspend')
        ssh.close()
        logger.info("Rig suspended to RAM")
    except ImportError:
        # No paramiko — use raw socket to trigger suspend via llama-server shutdown
        logger.warning("paramiko not installed — cannot suspend rig remotely")
    except Exception as e:
        logger.warning("Could not suspend rig: %s", e)


def rig_shutdown():
    """Full shutdown. Models lost, needs full reload on next wake."""
    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(RIG_IP, username='user', password='1', timeout=5)
        ssh.exec_command('echo 1 | sudo -S shutdown -h now')
        ssh.close()
        logger.info("Rig shutting down")
    except Exception as e:
        logger.warning("Could not shut down rig: %s", e)


# ── Convenience: run directly to check rig status ─────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if "--wake" in sys.argv:
        status = ensure_rig_ready()
        if status:
            print(f"\nRig ready: {status['ready_count']}/{status['total']} GPUs")
            for port in status['ready']:
                print(f"  http://{RIG_IP}:{port}/v1/chat/completions")
        else:
            print("\nRig unavailable")
            sys.exit(1)

    elif "--sleep" in sys.argv:
        rig_sleep()

    elif "--shutdown" in sys.argv:
        rig_shutdown()

    elif "--status" in sys.argv:
        if not is_rig_awake():
            print("Rig: SLEEPING/OFF")
        else:
            status = get_gpu_status()
            print(f"Rig: AWAKE — {status['ready_count']}/{status['total']} GPUs ready")
            for port in status['ready']:
                print(f"  GPU port {port}: UP")
            for port in status['down']:
                print(f"  GPU port {port}: DOWN")

    else:
        print("AI Rig Client")
        print(f"  Rig IP:    {RIG_IP}")
        print(f"  GPU ports: {RIG_PORTS}")
        print()
        print("Commands:")
        print("  python rig_client.py --status    Check rig status")
        print("  python rig_client.py --wake      Wake rig + wait for GPUs")
        print("  python rig_client.py --sleep     Suspend to RAM (models stay)")
        print("  python rig_client.py --shutdown  Full power off")
        print()
        print("In your code:")
        print("  from rig_client import ensure_rig_ready, rig_sleep")
        print("  status = ensure_rig_ready()  # wakes + waits")
        print("  # ... send work to status['ready'] ports ...")
        print("  rig_sleep()  # suspend when done")

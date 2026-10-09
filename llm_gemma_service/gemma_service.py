"""Reuse or start the existing CUDA 12.2 Gemma server on a DataSphere VM.

No third-party Python dependencies. Provision the binary and weights using
the setup notebook first. Changing/reimporting this file never restarts a
healthy server. CLI: python gemma_service.py ensure|status|stop.
"""
import argparse
import contextlib
import fcntl
import json
import os
import platform
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(os.environ.get("GEMMA_ROOT", "/home/jupyter/project/llm/cuda122"))
PORT = int(os.environ.get("GEMMA_PORT", "11435"))
BASE = f"http://127.0.0.1:{PORT}"
MODEL_PATH = ROOT / "weights/gemma-4-26B_q4_0-it.gguf"
SERVER = ROOT / "build/bin/llama-server"
MAMBA = ROOT / "bin/micromamba"
TOOLCHAIN = ROOT / "toolchain"
SERVER_LOG = ROOT / "server.log"
STATE_FILE = ROOT / f"service-{PORT}.json"
LOCK_FILE = ROOT / f"service-{PORT}.lock"
OPENER = build_opener(ProxyHandler({}))


def api(path, payload=None, timeout=10):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        # Surface the server error body instead of losing it in urllib's message.
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {error.code}: {detail}") from error


def _ready():
    try:
        health = api("/health", timeout=2)
        if health.get("status") != "ok":
            return False
        models = api("/v1/models", timeout=2)
    except (RuntimeError, URLError, TimeoutError, ConnectionError):
        return False
    if "gemma" not in {item.get("id") for item in models.get("data", [])}:
        raise RuntimeError(f"На {BASE} работает другая модель. Чужой сервер не перезапускаем.")
    return True


def _identity(pid):
    """Boot ID + process start time avoid signalling an unrelated reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": stat[19]}
    except (OSError, IndexError):
        return None


def _read_state():
    try:
        state = json.loads(STATE_FILE.read_text())
        if int(state["pid"]) <= 1 or state["base_url"] != BASE:
            return None
        return state
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _owned_alive(state):
    if not state or not state.get("identity") or _identity(int(state["pid"])) != state["identity"]:
        return False
    try:
        return os.getpgid(int(state["pid"])) == int(state["pid"])
    except (ProcessLookupError, PermissionError):
        return False


@contextlib.contextmanager
def _locked(timeout):
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Другой процесс уже запускает/останавливает LLM. Повторите проверку позже.")
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save_state(state):
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.chmod(0o600)
    temporary.replace(STATE_FILE)


def log_tail():
    try:
        return "\n".join(SERVER_LOG.read_text(errors="replace").splitlines()[-60:])
    except OSError:
        return "Лог запуска отсутствует."


def _gpu_counts():
    try:
        text = SERVER_LOG.read_text(errors="replace")
    except OSError:
        return None
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    matches = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", text)
    return tuple(map(int, matches[-1])) if matches else None


def _check_assets():
    if platform.system() != "Linux":
        raise RuntimeError("Запускайте сервер внутри Linux-ВМ DataSphere.")
    if not SERVER.is_file() or not MAMBA.is_file() or not TOOLCHAIN.is_dir():
        raise RuntimeError("Сначала выполните ячейку 1: сборка CUDA 12.2/llama.cpp.")
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size <= 10 * 1024**3:
        raise RuntimeError("Сначала выполните ячейку 2: скачивание полных весов GGUF.")
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
    ], text=True, timeout=10).splitlines()
    if not devices or "L4" not in devices[0] or devices[0].split(",")[-1].strip() != "8.9":
        raise RuntimeError("Существующая сборка рассчитана на NVIDIA L4 (SM 8.9).")


def _launch():
    _check_assets()
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", PORT))
        except OSError as error:
            raise RuntimeError(f"Порт {PORT} занят, но совместимый LLM API не готов. "
                               "Чужой процесс не останавливаем.") from error
    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = str(ROOT / "mamba-cache")
    env["LD_LIBRARY_PATH"] = ":".join([
        str(ROOT / "build/bin"), str(TOOLCHAIN / "lib"),
        str(TOOLCHAIN / "targets/x86_64-linux/lib"), env.get("LD_LIBRARY_PATH", ""),
    ])
    command = [str(MAMBA), "run", "-p", str(TOOLCHAIN), str(SERVER),
               "--model", str(MODEL_PATH), "--alias", "gemma", "--host", "127.0.0.1",
               "--port", str(PORT), "--device", "CUDA0", "--n-gpu-layers", "all", "--fit", "off",
               "--ctx-size", "2048", "--parallel", "1", "--batch-size", "128", "--ubatch-size", "128",
               "--split-mode", "none", "--reasoning", "off", "--verbosity", "4", "--log-colors", "off"]
    with SERVER_LOG.open("w") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    identity = _identity(process.pid)
    if identity is None or process.poll() is not None:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        raise RuntimeError("Сервер сразу завершился:\n" + log_tail())
    state = {"pid": process.pid, "identity": identity, "base_url": BASE,
             "model_path": str(MODEL_PATH), "started_at": time.time(), "gpu_verified": False}
    try:
        _save_state(state)
    except Exception:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        raise
    return state, process


def _stop_owned(state):
    if not _owned_alive(state):
        return False
    pid = int(state["pid"])
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while _owned_alive(state) and time.monotonic() < deadline:
        time.sleep(0.2)
    if _owned_alive(state):
        os.killpg(pid, signal.SIGKILL)
    STATE_FILE.unlink(missing_ok=True)
    return True


def _wait_ready(state, timeout, process=None):
    started = time.perf_counter()
    report_at = 15
    while time.perf_counter() - started < timeout:
        if not _owned_alive(state):
            if process is not None:
                process.poll()  # Reap an exited child when this call started it.
            raise RuntimeError("Сервер завершился при загрузке:\n" + log_tail())
        if _ready():
            counts = _gpu_counts()
            if counts is None:
                time.sleep(0.1)  # Log output is asynchronous.
                continue
            loaded, total = counts
            if loaded <= 0 or loaded != total:
                raise RuntimeError(f"На GPU выгружено только {loaded}/{total} слоёв:\n" + log_tail())
            state["gpu_verified"] = True
            _save_state(state)
            return time.perf_counter() - started
        elapsed = time.perf_counter() - started
        if elapsed >= report_at:
            print(f"Загрузка и прогрев: {elapsed:.0f} с...", flush=True)
            report_at = elapsed + 15
        time.sleep(0.5)
    raise TimeoutError(f"LLM не готова или нет подтверждения GPU. Лог: {SERVER_LOG}\n" + log_tail())


def ensure_running(timeout=900):
    """Return immediately for a ready server; never restart it automatically."""
    started = time.perf_counter()
    if _ready():
        state = _read_state()
        # A previously interrupted startup still needs its GPU check.
        if not _owned_alive(state) or state.get("gpu_verified"):
            print("LLM уже работает. Веса повторно не загружаются.")
            return {"reused": True, "base_url": BASE, "wait_seconds": time.perf_counter() - started}
    with _locked(timeout):
        state = _read_state()
        if _ready() and (not _owned_alive(state) or state.get("gpu_verified")):
            print("LLM уже работает. Веса повторно не загружаются.")
            return {"reused": True, "base_url": BASE, "wait_seconds": time.perf_counter() - started}
        launched = False
        process = None
        if not _owned_alive(state):
            state, process = _launch()
            launched = True
            print(f"Запускаю LLM на {BASE}. Процесс и лог сохраняются независимо от ячейки.", flush=True)
        else:
            print("LLM уже загружается. Жду тот же процесс; вторую копию не запускаю.", flush=True)
        try:
            _wait_ready(state, timeout, process)
        except KeyboardInterrupt:
            print("Ожидание прервано. Сервер продолжает запускаться; повторите ensure_running().", flush=True)
            raise
        except Exception:
            # Only clean up a failed launch owned by this call.
            if launched:
                _stop_owned(state)
                if process is not None:
                    process.wait(timeout=30)
            raise
        elapsed = time.perf_counter() - started
        print(f"LLM готова. Все слои на GPU. Ожидание: {elapsed:.2f} с.", flush=True)
        return {"reused": not launched, "base_url": BASE, "wait_seconds": elapsed,
                "gpu_layers": _gpu_counts()}


def status():
    state = _read_state()
    return {"ready": _ready(), "managed_process_alive": _owned_alive(state),
            "base_url": BASE, "pid": state["pid"] if _owned_alive(state) else None,
            "log": str(SERVER_LOG)}


def stop_model():
    """Explicitly stop only a process started by this launcher, including across imports."""
    with _locked(35):
        state = _read_state()
        stopped = _stop_owned(state)
        if stopped:
            print("LLM остановлена. Веса на диске сохранены.")
        elif _ready():
            print("Сервер работает, но запущен не этим файлом. "
                  "Остановите его через старую ячейку stop_model(); чужой процесс не трогаем.")
        else:
            print("Управляемый LLM-процесс отсутствует.")
        return stopped


def measure_chat(messages, max_tokens=512, temperature=0.0, timeout=300):
    """Send role-preserving chat messages; return text, usage and request timing.

    Startup/reconnection time is reported separately in ``startup``. Imports
    and subsequent calls reuse the healthy server without reloading weights.
    """
    startup = ensure_running()
    started = time.perf_counter()
    response = api("/v1/chat/completions", {
        "model": "gemma", "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }, timeout=timeout)
    seconds = time.perf_counter() - started
    text = response["choices"][0]["message"]["content"]
    return {"text": text, "seconds": seconds, "startup": startup, "usage": response.get("usage")}


def measure_request(prompt, max_tokens=96, temperature=0.2, timeout=300):
    """Convenience wrapper compatible with the original notebook interface."""
    result = measure_chat([{"role": "user", "content": prompt}],
                          max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    print(result["text"])
    print(f"\nВремя запроса до полного ответа: {result['seconds']:.2f} с")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["ensure", "status", "stop"], nargs="?", default="ensure")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.action == "ensure":
        print(json.dumps(ensure_running(args.timeout), ensure_ascii=False, indent=2))
    elif args.action == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    else:
        stop_model()

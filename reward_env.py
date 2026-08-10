import os
import re
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any


class RewardEnvError(Exception):
    """Base exception for all RewardEnv errors."""
    pass


class ToolchainError(RewardEnvError):
    """Raised when compiler, linker, QEMU, or plugin binary is missing or invalid."""
    pass


class CompilationError(RewardEnvError):
    """Raised when assembly compilation or linking fails."""
    pass


class AssemblyError(CompilationError):
    """Raised when assembler fails."""
    pass


class LinkError(CompilationError):
    """Raised when linker fails."""
    pass


class ExecutionError(RewardEnvError):
    """Raised when QEMU execution fails or times out."""
    pass


class QEMUExecutionError(ExecutionError):
    """Raised when QEMU execution fails or times out."""
    pass


class MeasurementError(RewardEnvError):
    """Raised when reading or parsing execution cost fails."""
    pass


class PluginOutputError(MeasurementError):
    """Raised when plugin output is missing or invalid JSON."""
    pass


class RewardEnv:
    def __init__(
        self,
        compiler: Optional[str] = None,
        assembler: Optional[str] = None,
        linker: Optional[str] = None,
        qemu_binary: Optional[str] = None,
        plugin_path: Optional[str] = None,
        timeout_seconds: float = 5.0,
        strict: bool = True,
        require_start: bool = True,
        penalty_value: float = -1000000.0,
    ):
        self.timeout_seconds = timeout_seconds
        self.timeout = timeout_seconds
        self.strict = strict
        self.require_start = require_start
        self.penalty_value = penalty_value
        self.failure_penalty = penalty_value
        self.baseline_cycle_count: Optional[int] = None

        # Toolchain Detection
        self.compiler = self._detect(
            compiler or assembler,
            "RISCV_AS",
            [
                "riscv64-linux-gnu-gcc",
                "riscv64-linux-gnu-as",
                "riscv64-unknown-linux-gnu-gcc",
                "riscv64-unknown-linux-gnu-as",
            ],
        )
        self.as_bin = self.compiler
        self.ld_bin = self._detect_linker(linker)
        self.qemu_binary = self._detect_qemu(qemu_binary)
        self.qemu_bin = self.qemu_binary
        self.plugin_path = self._detect_plugin(plugin_path)

        self._validate_toolchain()

    def _detect(self, arg: Optional[str], env_var: str, defaults: list[str]) -> Optional[str]:
        if arg:
            w = shutil.which(arg)
            if w:
                return w
            if Path(arg).exists():
                return str(Path(arg).absolute())
            raise ToolchainError(f"Tool not found: {arg}")
        if os.getenv(env_var):
            env_val = os.getenv(env_var)
            w = shutil.which(env_val)
            if w:
                return w
            if Path(env_val).exists():
                return str(Path(env_val).absolute())
            raise ToolchainError(f"Tool specified in {env_var} not found: {env_val}")
        for d in defaults:
            w = shutil.which(d)
            if w:
                return w
            if Path(d).exists():
                return str(Path(d).absolute())
        return None

    def _detect_linker(self, arg: Optional[str]) -> Optional[str]:
        if arg:
            w = shutil.which(arg)
            if w:
                return w
            if Path(arg).exists():
                return str(Path(arg).absolute())
            raise ToolchainError(f"Linker not found: {arg}")
        if os.getenv("RISCV_LD"):
            env_val = os.getenv("RISCV_LD")
            w = shutil.which(env_val)
            if w:
                return w
            if Path(env_val).exists():
                return str(Path(env_val).absolute())
            raise ToolchainError(f"Linker specified in RISCV_LD not found: {env_val}")
        defaults = ["riscv64-linux-gnu-ld", "riscv64-unknown-linux-gnu-ld"]
        for d in defaults:
            w = shutil.which(d)
            if w:
                return w
            if Path(d).exists():
                return str(Path(d).absolute())
        return None

    def _detect_qemu(self, arg: Optional[str]) -> Optional[str]:
        if arg:
            w = shutil.which(arg)
            if w:
                return w
            p = Path(arg)
            if p.exists():
                return str(p.absolute())
            raise ToolchainError(f"QEMU binary not found: {arg}")
        if os.getenv("QEMU_RISCV64"):
            val = os.getenv("QEMU_RISCV64")
            w = shutil.which(val)
            if w:
                return w
            p = Path(val)
            if p.exists():
                return str(p.absolute())
            raise ToolchainError(f"QEMU binary specified in QEMU_RISCV64 not found: {val}")
        local_qemu = Path("./qemu/build/qemu-riscv64")
        if local_qemu.exists():
            return str(local_qemu.absolute())
        w = shutil.which("qemu-riscv64")
        if w:
            return w
        return None

    def _detect_plugin(self, arg: Optional[str]) -> Optional[str]:
        if arg:
            p = Path(arg)
            if not p.exists():
                raise ToolchainError(f"Plugin path does not exist: {arg}")
            return str(p.absolute())
        if os.getenv("CYCLE_COUNTER_PLUGIN"):
            val = os.getenv("CYCLE_COUNTER_PLUGIN")
            p = Path(val)
            if not p.exists():
                raise ToolchainError(f"Plugin path specified in CYCLE_COUNTER_PLUGIN does not exist: {val}")
            return str(p.absolute())
        local_plugin = Path("./cycle_counter.so")
        if local_plugin.exists():
            return str(local_plugin.absolute())
        return None

    def _validate_toolchain(self) -> None:
        tools = {
            "Compiler": self.compiler,
            "QEMU": self.qemu_binary,
            "Plugin": self.plugin_path,
        }
        for name, path in tools.items():
            if not path:
                raise ToolchainError(f"{name} not found. Check arguments or environment variables.")

    def assemble(self, source_path: Path, object_path: Path) -> None:
        cmd = [self.compiler, "-march=rv64gc", "-mabi=lp64d", "-c", "-o", str(object_path), str(source_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as e:
            raise CompilationError("Assembly timed out") from e
        if res.returncode != 0:
            raise AssemblyError(res.stderr or f"Assembler exited with code {res.returncode}")

    def link(self, object_path: Path, binary_path: Path) -> None:
        if self.ld_bin and "ld" in self.ld_bin:
            cmd = [self.ld_bin, "-m", "elf64lriscv", "-Ttext=0x10000", "-o", str(binary_path), str(object_path)]
        else:
            cmd = [self.compiler, "-march=rv64gc", "-mabi=lp64d", "-nostdlib", "-static", "-o", str(binary_path), str(object_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as e:
            raise CompilationError("Linking timed out") from e
        if res.returncode != 0:
            raise LinkError(res.stderr or f"Linker exited with code {res.returncode}")

    def compile_asm(self, asm_text: str, output_path: Union[str, Path]) -> None:
        if asm_text is None or not isinstance(asm_text, str):
            raise CompilationError("Assembly text must be a non-null string")
        if not asm_text.strip():
            raise CompilationError("Assembly text is empty")
        if self.require_start and "_start" not in asm_text:
            raise CompilationError("Assembly text does not contain '_start' symbol")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_s = Path(tmp_dir) / "code.s"
            tmp_s.write_text(asm_text)

            if "gcc" in self.compiler:
                cmd = [
                    self.compiler,
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-nostdlib",
                    "-static",
                    "-o",
                    str(output_path),
                    str(tmp_s),
                ]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired as e:
                    raise CompilationError("Compilation timed out") from e
                if res.returncode != 0:
                    raise CompilationError(res.stderr or f"Compiler exited with code {res.returncode}")
            else:
                tmp_o = Path(tmp_dir) / "code.o"
                self.assemble(tmp_s, tmp_o)
                self.link(tmp_o, output_path)

    def run_binary(
        self,
        binary_path: Union[str, Path],
        json_output_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        binary_path = Path(binary_path)
        if not binary_path.exists():
            raise ExecutionError(f"Binary file not found: {binary_path}")

        if json_output_path:
            json_out = Path(json_output_path)
            json_out.parent.mkdir(parents=True, exist_ok=True)
            plugin_arg = f"{self.plugin_path},output={json_out}"
        else:
            json_out = None
            plugin_arg = f"{self.plugin_path},stdout=1"

        cmd = [self.qemu_binary, "-plugin", plugin_arg, str(binary_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as e:
            raise QEMUExecutionError("QEMU execution timed out") from e

        if res.returncode != 0:
            raise QEMUExecutionError(f"QEMU exited with code {res.returncode}: {res.stderr}")

        if json_out and json_out.exists():
            output_str = json_out.read_text()
        else:
            output_str = res.stdout

        return self.parse_plugin_output(output_str)

    def parse_plugin_output(self, stdout: str) -> dict:
        if not stdout or not isinstance(stdout, str):
            raise PluginOutputError("QEMU plugin output is empty or invalid")

        start_idx = stdout.find("{")
        if start_idx != -1:
            try:
                decoder = json.JSONDecoder()
                data, _ = decoder.raw_decode(stdout[start_idx:])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\"weighted_cost\".*\}", stdout, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        raise PluginOutputError("Could not find JSON report with 'weighted_cost' in QEMU output")

    def parse_weighted_cost(self, stdout: str) -> float:
        data = self.parse_plugin_output(stdout)
        if "weighted_cost" not in data:
            raise PluginOutputError("JSON output missing 'weighted_cost' field")
        return float(data["weighted_cost"])

    def compile_and_run(self, asm_text: str) -> int:
        if not self.strict:
            try:
                return self._compile_and_run_impl(asm_text)
            except Exception:
                return int(self.penalty_value)
        else:
            return self._compile_and_run_impl(asm_text)

    def _compile_and_run_impl(self, asm_text: str) -> int:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            elf_file = base_path / "code.elf"
            json_file = base_path / "result.json"

            self.compile_asm(asm_text, elf_file)
            data = self.run_binary(elf_file, json_file)
            if "weighted_cost" not in data:
                raise PluginOutputError("Missing weighted_cost in plugin output")
            return int(data["weighted_cost"])

    def calculate_reward(
        self,
        cycle_count: Union[int, float],
        baseline_cycle_count: Union[int, float],
    ) -> float:
        return float(baseline_cycle_count - cycle_count)

    def reset(self) -> dict:
        self.baseline_cycle_count = None
        return {"cycle_count": None}

    def step(self, action_asm: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        cycle_count = self.compile_and_run(action_asm)
        if self.baseline_cycle_count is None:
            self.baseline_cycle_count = cycle_count
            reward = 0.0
        else:
            reward = self.calculate_reward(cycle_count, self.baseline_cycle_count)

        obs = {"cycle_count": cycle_count}
        done = True
        info = {}
        return obs, reward, done, info

    def __repr__(self) -> str:
        return (
            f"RewardEnv(compiler={self.compiler!r}, "
            f"qemu={self.qemu_binary!r}, "
            f"plugin={self.plugin_path!r}, "
            f"strict={self.strict})"
        )

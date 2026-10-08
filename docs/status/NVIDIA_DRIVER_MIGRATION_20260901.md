# NVIDIA Driver Migration Record (2026-09-01)

## Purpose

Move the RTX 4090 host from NVIDIA open driver `595.84` to the supported 580 branch so Isaac Sim 5.1 can start RoboDojo without the confirmed `librtx.scenedb.plugin.so` crash.

## Preflight

- OS: Ubuntu 24.04.4 LTS
- Running kernel: `7.0.0-30-generic`
- Previous driver: `nvidia-driver-595-open 595.84-0ubuntu0.24.04.1`
- Target driver: `nvidia-driver-580-open 580.173.02-0ubuntu0.24.04.1`
- Matching precompiled module: `linux-modules-nvidia-580-open-7.0.0-30-generic`
- Matching kernel headers: installed
- Secure Boot: disabled
- Free root space before migration: approximately 75 GB
- Fallback kernels retained: `7.0.0-28-generic`, `7.0.0-30-generic`
- The apt simulation replaces 17 NVIDIA 595 packages with the corresponding 580 packages. It does not remove CUDA Conda environments, Docker, model weights, or project files.

## Services Stopped Before Migration

The following user services used the RTX 4090 and were stopped before replacing driver libraries:

```text
g2-agent-asr.service
g2-agent-vlm.service
g2-agent-phase1.service
g2-agent-scene-announcer.service
g2-collection-hub-lan.service
g2a-head-av-tunnel.service
```

They can be restarted after driver validation with:

```bash
systemctl --user start \
  g2-agent-asr.service \
  g2-agent-vlm.service \
  g2-agent-phase1.service \
  g2-agent-scene-announcer.service \
  g2-collection-hub-lan.service \
  g2a-head-av-tunnel.service
```

The RoboDojo smoke test has completed. Keep these services stopped while subsequent CARVE/StarVLA evaluations need the GPU; they may otherwise be restarted with the command above.

## Installation Result

- `nvidia-driver-580-open 580.173.02-0ubuntu0.24.04.1`: installed
- `linux-modules-nvidia-580-open-7.0.0-30-generic`: installed
- `nvidia-dkms-580-open 580.173.02`: installed
- DKMS build completed; its module version exactly matched Ubuntu's precompiled module.
- `modprobe --show-depends` resolves `nvidia`, `nvidia-modeset`, and `nvidia-drm` to `/lib/modules/7.0.0-30-generic/kernel/nvidia-580-open/`.
- Module vermagic matches `7.0.0-30-generic` and is signed by Canonical.
- initramfs and GRUB were regenerated successfully; both fallback kernels remain present.
- `dpkg --audit` reported no incomplete packages.
- The required reboot completed successfully.

## Post-reboot Result

- Running kernel: `7.0.0-30-generic`
- Loaded NVIDIA driver: `580.173.02`
- `nvidia-smi`: passed
- StarVLA PyTorch CUDA and FlashAttention kernel test: passed
- RoboDojo PyTorch CUDA test: passed
- NVIDIA CDI refresh and GPU enumeration: passed
- Isaac Sim 5.1 startup and shutdown: passed without the previous exit-139 RTX crash
- Official RoboDojo doctor: `18 passed, 1 warning, 0 failed`
- Full one-environment StarVLA + RoboDojo episode: completed for all 1,050 control steps
- Failed system units after evaluation: 0
- Failed user units after evaluation: 0

The driver migration is validated. The one baseline episode did not solve `build_tower`, but it produced a valid result record and three 1,051-frame videos; this is a policy outcome, not a driver or simulator failure.

## Installation Command

```bash
sudo apt-get install -y \
  nvidia-driver-580-open \
  linux-modules-nvidia-580-open-7.0.0-30-generic
```

## Post-install Checks Before Reboot

```bash
dpkg -l | rg 'nvidia-driver-580-open|linux-modules-nvidia-580-open-7.0.0-30'
modinfo -F version /lib/modules/7.0.0-30-generic/updates/dkms/nvidia.ko 2>/dev/null \
  || modinfo -F version nvidia
lsinitramfs /boot/initrd.img-7.0.0-30-generic | rg 'nvidia.*\.ko' | head
```

The currently loaded module remains 595 until reboot. A userspace/kernel version mismatch before reboot is expected, so CUDA experiments must remain stopped during this interval.

## Rollback From a Text Console

If the graphical desktop does not start, enter a TTY with `Ctrl+Alt+F3`, log in, and run:

```bash
sudo apt-get install -y \
  nvidia-driver-595-open=595.84-0ubuntu0.24.04.1 \
  linux-modules-nvidia-595-open-7.0.0-30-generic=7.0.0-30.30~24.04.1
sudo update-initramfs -u -k 7.0.0-30-generic
sudo reboot
```

If necessary, GRUB also retains the `7.0.0-28-generic` kernel under Advanced options for Ubuntu.

## Completed Post-reboot Validation

1. `nvidia-smi` reports the RTX 4090 and driver `580.173.02`: passed.
2. PyTorch CUDA tests in both `StarVLA` and `RoboDojo`: passed.
3. Isaac Sim reaches `Simulation App Startup Complete` without exit 139: passed.
4. The official RoboDojo doctor reports zero failures: passed.
5. One-environment, one-episode StarVLA + RoboDojo smoke test: passed as an execution-path test.

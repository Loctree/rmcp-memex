//! Live system monitor for the indexing dashboard.

use std::ffi::{OsStr, OsString};
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessesToUpdate, System};
use tokio::sync::watch;
use tokio::task::JoinHandle;

const GPU_CLASSES: &[&str] = &["AGXAcceleratorG15X", "IOAccelerator"];
const EMBEDDER_PROCESS_NAMES: &[&str] = &["ollama", "llama-server", "mlx_server", "mlx-server"];

/// GPU probe status for the dashboard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuStatus {
    Available { class_name: String },
    Unavailable { reason: String },
}

impl Default for GpuStatus {
    fn default() -> Self {
        Self::Unavailable {
            reason: "not sampled yet".to_string(),
        }
    }
}

/// Latest system monitor snapshot for the dashboard.
#[derive(Debug, Clone)]
pub struct MonitorSnapshot {
    pub system_cpu_percent: f32,
    pub system_ram_used: u64,
    pub system_ram_total: u64,
    pub rmcp_memex_cpu: f32,
    pub rmcp_memex_rss: u64,
    pub embedder_cpu_aggregate: f32,
    pub embedder_rss_aggregate: u64,
    pub gpu_util_percent: Option<f32>,
    pub gpu_memory_used: Option<u64>,
    pub gpu_memory_total: Option<u64>,
    pub gpu_status: GpuStatus,
    pub sampled_at: Instant,
}

impl Default for MonitorSnapshot {
    fn default() -> Self {
        Self {
            system_cpu_percent: 0.0,
            system_ram_used: 0,
            system_ram_total: 0,
            rmcp_memex_cpu: 0.0,
            rmcp_memex_rss: 0,
            embedder_cpu_aggregate: 0.0,
            embedder_rss_aggregate: 0,
            gpu_util_percent: None,
            gpu_memory_used: None,
            gpu_memory_total: None,
            gpu_status: GpuStatus::default(),
            sampled_at: Instant::now(),
        }
    }
}

impl MonitorSnapshot {
    pub fn format_bytes(bytes: u64) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;
        const GB: f64 = MB * 1024.0;

        match bytes {
            0..=1023 => format!("{bytes} B"),
            1_024..=1_048_575 => format!("{:.0} KB", bytes as f64 / KB),
            1_048_576..=1_073_741_823 => format!("{:.0} MB", bytes as f64 / MB),
            _ => format!("{:.1} GB", bytes as f64 / GB),
        }
    }
}

/// Spawn the system monitor sampler with a latest-value watch channel.
pub fn spawn_monitor(interval: Duration) -> (watch::Receiver<MonitorSnapshot>, JoinHandle<()>) {
    let (sender, receiver) = watch::channel(MonitorSnapshot::default());
    let handle = tokio::spawn(async move {
        let my_pid = Pid::from_u32(std::process::id());
        let mut system = System::new_all();
        system.refresh_all();
        tokio::time::sleep(Duration::from_millis(250)).await;

        loop {
            system.refresh_cpu_usage();
            system.refresh_memory();
            system.refresh_processes(ProcessesToUpdate::All, true);

            let snapshot = build_snapshot(&system, my_pid);
            if sender.send(snapshot).is_err() {
                break;
            }
            tokio::time::sleep(interval).await;
        }
    });

    (receiver, handle)
}

fn build_snapshot(system: &System, my_pid: Pid) -> MonitorSnapshot {
    let mut snapshot = MonitorSnapshot {
        system_cpu_percent: system.global_cpu_usage(),
        system_ram_used: system.used_memory(),
        system_ram_total: system.total_memory(),
        sampled_at: Instant::now(),
        ..MonitorSnapshot::default()
    };

    if let Some(process) = system.process(my_pid) {
        snapshot.rmcp_memex_cpu = process.cpu_usage();
        snapshot.rmcp_memex_rss = process.memory();
    }

    for process in system.processes().values() {
        if is_embedder_process(process.name(), process.cmd()) {
            snapshot.embedder_cpu_aggregate += process.cpu_usage();
            snapshot.embedder_rss_aggregate += process.memory();
        }
    }

    match probe_gpu() {
        Ok(metrics) => {
            snapshot.gpu_util_percent = Some(metrics.device_util as f32);
            snapshot.gpu_memory_used = metrics.memory_used;
            snapshot.gpu_memory_total = metrics.memory_total;
            snapshot.gpu_status = GpuStatus::Available {
                class_name: metrics.class_name,
            };
        }
        Err(status) => {
            snapshot.gpu_status = status;
        }
    }

    snapshot
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuMetrics {
    class_name: String,
    device_util: u64,
    memory_used: Option<u64>,
    memory_total: Option<u64>,
}

fn probe_gpu() -> Result<GpuMetrics, GpuStatus> {
    #[cfg(not(target_os = "macos"))]
    return Err(GpuStatus::Unavailable {
        reason: "GPU telemetry only supported on macOS via ioreg".to_string(),
    });

    let mut reasons = Vec::new();

    for class_name in GPU_CLASSES {
        match std::process::Command::new("ioreg")
            .args(["-l", "-w", "0", "-r", "-c", class_name, "-d", "1"])
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(metrics) = parse_ioreg_output(&stdout, class_name) {
                    return Ok(metrics);
                }
                reasons.push(format!("{class_name}: telemetry keys not found"));
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                if stderr.is_empty() {
                    reasons.push(format!("{class_name}: ioreg exited with {}", output.status));
                } else {
                    reasons.push(format!("{class_name}: {stderr}"));
                }
            }
            Err(error) => {
                reasons.push(format!("{class_name}: {error}"));
            }
        }
    }

    Err(GpuStatus::Unavailable {
        reason: reasons.join(" | "),
    })
}

fn parse_ioreg_output(output: &str, class_name: &str) -> Option<GpuMetrics> {
    let device_util = extract_ioreg_value(output, "Device Utilization %")
        .or_else(|| extract_ioreg_value(output, "Renderer Utilization %"))?;

    Some(GpuMetrics {
        class_name: class_name.to_string(),
        device_util,
        memory_used: extract_ioreg_value(output, "In use system memory"),
        memory_total: extract_ioreg_value(output, "Alloc system memory"),
    })
}

fn extract_ioreg_value(output: &str, key: &str) -> Option<u64> {
    let quoted_key = format!("\"{key}\"");
    let key_index = output.find(&quoted_key)?;
    let value_region = &output[key_index + quoted_key.len()..];
    let equals_index = value_region.find('=')?;
    let remainder = value_region[equals_index + 1..].trim_start();
    let digits: String = remainder
        .chars()
        .skip_while(|ch| !ch.is_ascii_digit())
        .take_while(|ch| ch.is_ascii_digit())
        .collect();

    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn is_embedder_process(name: &OsStr, cmdline: &[OsString]) -> bool {
    let name = name.to_string_lossy().to_lowercase();
    if EMBEDDER_PROCESS_NAMES
        .iter()
        .any(|candidate| name.contains(candidate))
    {
        return true;
    }

    if name.contains("python") {
        let cmdline = cmdline
            .iter()
            .map(|segment| segment.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        return cmdline.contains("mlx") || cmdline.contains("embed");
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ioreg_fixture_file() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ioreg_m3_ultra.txt");
        let fixture = std::fs::read_to_string(&fixture_path).expect("read ioreg fixture");
        let metrics = parse_ioreg_output(&fixture, "AGXAcceleratorG15X").expect("parse ioreg");
        assert_eq!(metrics.class_name, "AGXAcceleratorG15X");
        assert!(metrics.device_util <= 100);
        assert!(metrics.memory_total.is_some());
    }

    #[test]
    fn extract_ioreg_value_handles_inline_statistics() {
        let sample = "\"PerformanceStatistics\" = {\"Device Utilization %\"=13,\"Alloc system memory\"=477265838080,\"In use system memory\"=1874984960}";
        assert_eq!(
            extract_ioreg_value(sample, "Device Utilization %"),
            Some(13)
        );
        assert_eq!(
            extract_ioreg_value(sample, "Alloc system memory"),
            Some(477_265_838_080)
        );
        assert_eq!(
            extract_ioreg_value(sample, "In use system memory"),
            Some(1_874_984_960)
        );
    }

    #[test]
    fn embedder_process_detection_matches_expected_names() {
        assert!(is_embedder_process(OsStr::new("ollama"), &[]));
        assert!(is_embedder_process(OsStr::new("llama-server"), &[]));
        assert!(is_embedder_process(OsStr::new("mlx_server"), &[]));
        assert!(is_embedder_process(
            OsStr::new("python3"),
            &[
                OsString::from("python3"),
                OsString::from("-m"),
                OsString::from("mlx.embed")
            ]
        ));
        assert!(!is_embedder_process(OsStr::new("nginx"), &[]));
    }
}

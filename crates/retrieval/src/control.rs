use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

/// Control signals for generation — passed from the binary where the statics live.
pub struct GenerationControl {
    pub generating: Arc<AtomicBool>,
    pub sigint_received: Arc<AtomicBool>,
    pub gen_interrupted: Arc<AtomicBool>,
}

/// Animated spinner displayed while waiting for the first token from the LLM.
pub struct Spinner {
    pub alive: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    pub fn start() -> Self {
        let alive = Arc::new(AtomicBool::new(true));
        let a = alive.clone();
        let handle = thread::spawn(move || {
            let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let mut i = 0;
            while a.load(Ordering::Relaxed) {
                eprint!("\r\x1b[2K\x1b[90m{} réflexion...\x1b[0m", frames[i % frames.len()]);
                let _ = io::stderr().flush();
                i += 1;
                thread::sleep(std::time::Duration::from_millis(80));
            }
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
        });
        Self { alive, handle: Some(handle) }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

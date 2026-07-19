mod model;
mod protocol;

use std::io::{self, BufRead, Write};

use candle_core::Device;
use clap::Parser;

use crate::model::Embedder;
use crate::protocol::{Response, encode_response, parse_request};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    tokenizer: String,
}

fn main() {
    let args = Args::parse();

    let device = match select_device() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("selecting device: {e}");
            std::process::exit(1);
        }
    };

    let embedder = match Embedder::load(&args.model, &args.tokenizer, device) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load model/tokenizer: {e}");
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("reading stdin: {e}");
                break;
            }
        };
        if line.is_empty() {
            continue;
        }

        let response = handle_request(&embedder, &line);
        // A line is always written here, even if encoding response
        // fails: the caller does one blocking read per request, so
        // skipping a line instead of writing a fallback would hang it
        // rather than surface an error.
        let encoded = encode_response(&response).unwrap_or_else(|e| {
            eprintln!("encoding response: {e}");
            r#"{"error":"internal error: failed to encode response"}"#.to_string()
        });
        if writeln!(out, "{encoded}").is_err() || out.flush().is_err() {
            break;
        }
    }
}

/// Reads the CANDLE_DEVICE environment variable: "metal" constructs a
/// Metal device, unset keeps the CPU default. No CLI flag — the
/// sidecar adapter and both Go CLIs it's wired into need zero changes,
/// since os/exec already passes the parent process's environment
/// through to this spawned child.
fn select_device() -> candle_core::Result<Device> {
    match std::env::var("CANDLE_DEVICE") {
        Ok(v) if v == "metal" => Device::new_metal(0),
        Ok(v) => {
            // Set but not recognized (e.g. a typo like "Metal" or
            // "mps") — warn rather than silently falling back to CPU,
            // so a misconfiguration doesn't masquerade as a real CPU
            // measurement.
            eprintln!(
                "CANDLE_DEVICE={v:?} not recognized, falling back to CPU (only \"metal\" selects Metal)"
            );
            Ok(Device::Cpu)
        }
        Err(_) => Ok(Device::Cpu),
    }
}

fn handle_request(embedder: &Embedder, line: &str) -> Response {
    let request = match parse_request(line) {
        Ok(r) => r,
        Err(e) => return Response::error(format!("parsing request: {e}")),
    };

    match embedder.embed_batch(&request.chunks) {
        Ok(vectors) => Response::vectors(vectors),
        Err(e) => Response::error(e),
    }
}

//! ndjson request/response types for the sidecar's stdio protocol: one
//! JSON line in, one JSON line out, looping until stdin closes. Kept
//! separate from `main.rs` so it's unit-testable without a real model.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, PartialEq)]
pub struct Request {
    pub chunks: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Response {
    Vectors { vectors: Vec<Vec<f32>> },
    Error { error: String },
}

impl Response {
    pub fn vectors(vectors: Vec<Vec<f32>>) -> Self {
        Response::Vectors { vectors }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Response::Error {
            error: message.into(),
        }
    }
}

/// Parses one ndjson request line.
pub fn parse_request(line: &str) -> Result<Request, serde_json::Error> {
    serde_json::from_str(line)
}

/// Serializes a response as one ndjson line (no trailing newline —
/// callers write the newline when framing the line on the wire).
pub fn encode_response(response: &Response) -> Result<String, serde_json::Error> {
    serde_json::to_string(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_request_with_multiple_chunks() {
        let req = parse_request(r#"{"chunks": ["hello", "world"]}"#).unwrap();
        assert_eq!(
            req,
            Request {
                chunks: vec!["hello".to_string(), "world".to_string()],
            }
        );
    }

    #[test]
    fn parses_a_request_with_no_chunks() {
        let req = parse_request(r#"{"chunks": []}"#).unwrap();
        assert_eq!(req, Request { chunks: vec![] });
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_request("not json").is_err());
    }

    #[test]
    fn rejects_a_request_missing_the_chunks_field() {
        assert!(parse_request(r#"{"not_chunks": []}"#).is_err());
    }

    #[test]
    fn encodes_a_vectors_response() {
        let resp = Response::vectors(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        let line = encode_response(&resp).unwrap();
        assert_eq!(line, r#"{"vectors":[[0.1,0.2],[0.3,0.4]]}"#);
    }

    #[test]
    fn encodes_an_error_response() {
        let resp = Response::error("something went wrong");
        let line = encode_response(&resp).unwrap();
        assert_eq!(line, r#"{"error":"something went wrong"}"#);
    }

    #[test]
    fn round_trips_a_vectors_response_through_the_wire_format() {
        let original = Response::vectors(vec![vec![1.0, -2.5, 3.0]]);
        let line = encode_response(&original).unwrap();
        let decoded: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(decoded["vectors"][0][0], 1.0);
        assert_eq!(decoded["vectors"][0][1], -2.5);
        assert_eq!(decoded["vectors"][0][2], 3.0);
        assert!(decoded.get("error").is_none());
    }
}

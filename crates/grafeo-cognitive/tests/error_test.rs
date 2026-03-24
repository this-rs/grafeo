//! Tests for the CognitiveError type.

use grafeo_cognitive::CognitiveError;

#[test]
fn energy_error_display() {
    let err = CognitiveError::Energy("node not found".into());
    assert_eq!(err.to_string(), "energy error: node not found");
}

#[test]
fn synapse_error_display() {
    let err = CognitiveError::Synapse("self-synapse".into());
    assert_eq!(err.to_string(), "synapse error: self-synapse");
}

#[test]
fn fabric_error_display() {
    let err = CognitiveError::Fabric("metric overflow".into());
    assert_eq!(err.to_string(), "fabric error: metric overflow");
}

#[test]
fn config_error_display() {
    let err = CognitiveError::Config("invalid threshold".into());
    assert_eq!(err.to_string(), "config error: invalid threshold");
}

#[test]
fn store_error_display() {
    let err = CognitiveError::Store("not found".into());
    assert_eq!(err.to_string(), "store error: not found");
}

#[test]
fn reactive_error_from_conversion() {
    let reactive_err = grafeo_reactive::ReactiveError::BusCapacityExceeded(42);
    let cognitive_err: CognitiveError = reactive_err.into();
    let msg = cognitive_err.to_string();
    assert!(msg.starts_with("reactive error:"), "got: {msg}");
}

#[test]
fn error_is_debug() {
    let err = CognitiveError::Energy("test".into());
    let dbg = format!("{:?}", err);
    assert!(dbg.contains("Energy"), "got: {dbg}");
}

#[test]
fn cognitive_result_ok() {
    fn produce_ok() -> grafeo_cognitive::error::CognitiveResult<i32> {
        Ok(42)
    }
    let result = produce_ok();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn cognitive_result_err() {
    let result: grafeo_cognitive::error::CognitiveResult<i32> =
        Err(CognitiveError::Config("bad".into()));
    assert!(result.is_err());
}

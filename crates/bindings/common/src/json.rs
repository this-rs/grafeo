//! Bidirectional conversion between `serde_json::Value` and `grafeo Value`.
//!
//! Used by the Node.js and C bindings for parameter parsing and result
//! serialization. The C binding adds a thin wrapper for its `$timestamp_us`
//! convention on top of [`value_to_json`].

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::{PropertyKey, Value};

/// Convert a `serde_json::Value` to a Grafeo [`Value`].
///
/// Numbers are parsed as `Int64` when they fit, otherwise `Float64`.
/// Objects with a `$timestamp_us` key are decoded as timestamps.
pub fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.as_str().into()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::List(items.into())
        }
        serde_json::Value::Object(obj) => {
            // Check for special $timestamp_us encoding (used by C binding).
            if let Some(ts) = obj.get("$timestamp_us").and_then(serde_json::Value::as_i64) {
                return Value::Timestamp(grafeo_common::types::Timestamp::from_micros(ts));
            }
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(PropertyKey::new(k.clone()), json_to_value(v));
            }
            Value::Map(Arc::new(map))
        }
    }
}

/// Convert a Grafeo [`Value`] to a `serde_json::Value`.
///
/// Timestamps are encoded as `{ "$timestamp_us": <micros> }`.
/// Bytes are encoded as a JSON array of integers for lossless roundtrip.
/// Vectors are encoded as a JSON array of floats.
pub fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int64(i) => serde_json::json!(*i),
        Value::Float64(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::Bytes(b) => {
            let arr: Vec<serde_json::Value> =
                b.iter().map(|&byte| serde_json::json!(byte)).collect();
            serde_json::Value::Array(arr)
        }
        Value::Timestamp(ts) => serde_json::json!({ "$timestamp_us": ts.as_micros() }),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_json).collect()),
        Value::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Vector(vec) => {
            serde_json::Value::Array(vec.iter().map(|&f| serde_json::json!(f)).collect())
        }
    }
}

/// Convert a JSON params object to a `HashMap<String, Value>`.
///
/// Returns `None` if the input is `None`. Returns an error string if the
/// input is not a JSON object.
pub fn json_params_to_map(
    params: Option<&serde_json::Value>,
) -> Result<Option<std::collections::HashMap<String, Value>>, String> {
    let Some(params) = params else {
        return Ok(None);
    };
    let Some(obj) = params.as_object() else {
        return Err("params must be an object".into());
    };
    let mut map = std::collections::HashMap::with_capacity(obj.len());
    for (key, value) in obj {
        map.insert(key.clone(), json_to_value(value));
    }
    Ok(Some(map))
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::Timestamp;

    use super::*;

    #[test]
    fn roundtrip_primitives() {
        assert_eq!(json_to_value(&serde_json::json!(null)), Value::Null);
        assert_eq!(json_to_value(&serde_json::json!(true)), Value::Bool(true));
        assert_eq!(json_to_value(&serde_json::json!(42)), Value::Int64(42));
        assert_eq!(
            json_to_value(&serde_json::json!(3.14)),
            Value::Float64(3.14)
        );
        assert_eq!(
            json_to_value(&serde_json::json!("hello")),
            Value::String("hello".into())
        );
    }

    #[test]
    fn roundtrip_timestamp() {
        let ts = Value::Timestamp(Timestamp::from_micros(1_000_000));
        let json = value_to_json(&ts);
        let back = json_to_value(&json);
        assert_eq!(ts, back);
    }

    #[test]
    fn roundtrip_nested() {
        let json = serde_json::json!({"a": [1, 2, 3], "b": {"c": true}});
        let val = json_to_value(&json);
        let back = value_to_json(&val);
        assert_eq!(json, back);
    }

    #[test]
    fn params_conversion() {
        let params = serde_json::json!({"name": "Alice", "age": 30});
        let map = json_params_to_map(Some(&params)).unwrap().unwrap();
        assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(map.get("age"), Some(&Value::Int64(30)));
    }

    #[test]
    fn params_none() {
        assert!(json_params_to_map(None).unwrap().is_none());
    }

    #[test]
    fn params_not_object() {
        let params = serde_json::json!([1, 2, 3]);
        assert!(json_params_to_map(Some(&params)).is_err());
    }
}

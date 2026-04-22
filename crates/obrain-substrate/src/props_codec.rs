//! # Value ↔ PropertyValue codec (T17c Step 3b.1)
//!
//! Bridges [`obrain_common::types::Value`] (the public property-value
//! type used by the graph API) and [`crate::page::PropertyValue`] (the
//! 9-tag on-disk encoding used by [`crate::page::PropertyPage`]).
//!
//! ## Encoding strategy
//!
//! Natively representable variants ride the fast path without going
//! through serde:
//!
//! | `Value` variant            | `PropertyValue` tag                               |
//! |----------------------------|---------------------------------------------------|
//! | `Null`                     | `Null`                                            |
//! | `Bool(b)`                  | `Bool(b)`                                         |
//! | `Int64(i)`                 | `I64(i)`                                          |
//! | `Float64(f)`               | `F64(f)`                                          |
//! | `String(s)`                | `StringRef(heap_intern(s.as_bytes()))`            |
//! | `Bytes(b)`                 | `BytesRef(heap_intern([0x00, b..]))`              |
//! | `List(xs)` — all `Int64`   | `ArrI64(xs)`                                      |
//! | `List(xs)` — all `Float64` | `ArrF64(xs)`                                      |
//! | `List(xs)` — all `String`  | `ArrStringRef([heap_intern(x.as_bytes()) for x])` |
//!
//! Everything else (`Timestamp`, `Date`, `Time`, `Duration`,
//! `ZonedDatetime`, heterogeneous `List`, `Map`, `Path`) uses a
//! **bincode fallback** stored as a heap blob: the discriminator byte
//! `0x01` is prepended to the bincode payload, and the whole buffer is
//! interned as a `BytesRef`. Raw `Bytes` values use discriminator `0x00`
//! so the decoder can disambiguate without looking at the tag alone.
//!
//! `Vector(Arc<[f32]>)` is NOT handled here — vector writes are routed
//! to [`crate::vec_column_registry::VecColumnRegistry`] upstream in
//! [`crate::store::SubstrateStore::set_node_property`]. Calling the
//! encoder with a `Value::Vector` returns an error.
//!
//! ## Decoder invariants
//!
//! * `StringRef(h)` → the heap bytes MUST be valid UTF-8 (returns
//!   `SubstrateError::WalBadFrame` on decode failure).
//! * `BytesRef(h)` → first byte picks the discriminator:
//!   * `0x00` — raw bytes (payload is `bytes[1..]`).
//!   * `0x01` — bincode'd [`Value`] (decoded via [`bincode::serde`]).
//!   * anything else — hard error.
//! * `ArrStringRef(refs)` → each ref dereferences to valid UTF-8.
//!
//! The codec is stateless except for the heap — no interning cache is
//! kept between calls. Callers that churn the same strings (e.g.
//! bulk imports) should hoist a `FxHashMap<ArcStr, HeapRef>` up at the
//! call site.

use std::sync::Arc;

use arcstr::ArcStr;
use bincode::config::{Configuration, Fixint, LittleEndian};
use obrain_common::types::Value;

use crate::error::{SubstrateError, SubstrateResult};
use crate::page::{HeapRef, PropertyValue};
use crate::props_zone::PropsZone;

/// Heap-blob discriminator: raw `Value::Bytes` payload.
pub const BYTES_DISCRIMINATOR_RAW: u8 = 0x00;
/// Heap-blob discriminator: bincode-encoded [`Value`] fallback.
pub const BYTES_DISCRIMINATOR_BINCODE: u8 = 0x01;

/// Matches the codec used by [`crate::props_snapshot`] so any future
/// migration between the sidecar and the zone doesn't re-encode.
fn bincode_config() -> Configuration<LittleEndian, Fixint> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

/// Convert a [`Value`] into a [`PropertyValue`], heap-interning any
/// variable-width payload on `zone`.
///
/// Returns `SubstrateError::WalBadFrame` for:
/// * `Value::Vector(_)` — must be routed to `vec_columns` upstream
/// * bincode encode failure (rare — implies a serde contract break)
/// * oversized heap payload (> one heap page minus 4 B length prefix)
pub fn encode_value(zone: &mut PropsZone, v: &Value) -> SubstrateResult<PropertyValue> {
    match v {
        Value::Null => Ok(PropertyValue::Null),
        Value::Bool(b) => Ok(PropertyValue::Bool(*b)),
        Value::Int64(i) => Ok(PropertyValue::I64(*i)),
        Value::Float64(f) => Ok(PropertyValue::F64(*f)),
        Value::String(s) => {
            let href = zone.intern_bytes(s.as_bytes())?;
            Ok(PropertyValue::StringRef(href))
        }
        Value::Bytes(b) => {
            let mut prefixed = Vec::with_capacity(b.len() + 1);
            prefixed.push(BYTES_DISCRIMINATOR_RAW);
            prefixed.extend_from_slice(b);
            let href = zone.intern_bytes(&prefixed)?;
            Ok(PropertyValue::BytesRef(href))
        }
        Value::List(xs) => encode_list(zone, xs, v),
        Value::Vector(_) => Err(SubstrateError::WalBadFrame(
            "Value::Vector must be routed to vec_columns upstream, not the props codec".into(),
        )),
        // Timestamp / Date / Time / Duration / ZonedDatetime / Map / Path
        // all share the bincode fallback path.
        other => encode_bincode_fallback(zone, other),
    }
}

fn encode_list(
    zone: &mut PropsZone,
    xs: &[Value],
    original: &Value,
) -> SubstrateResult<PropertyValue> {
    // Empty list → empty ArrI64 by convention. The decoder reverses
    // this by yielding `Value::List([])`, which is unambiguous since
    // element-less lists carry no type information anyway.
    if xs.is_empty() {
        return Ok(PropertyValue::ArrI64(Vec::new()));
    }
    if xs.iter().all(|v| matches!(v, Value::Int64(_))) {
        let ints: Vec<i64> = xs
            .iter()
            .filter_map(Value::as_int64)
            .collect();
        debug_assert_eq!(ints.len(), xs.len());
        return Ok(PropertyValue::ArrI64(ints));
    }
    if xs.iter().all(|v| matches!(v, Value::Float64(_))) {
        let floats: Vec<f64> = xs
            .iter()
            .filter_map(Value::as_float64)
            .collect();
        debug_assert_eq!(floats.len(), xs.len());
        return Ok(PropertyValue::ArrF64(floats));
    }
    if xs.iter().all(|v| matches!(v, Value::String(_))) {
        let mut refs = Vec::with_capacity(xs.len());
        for v in xs {
            if let Value::String(s) = v {
                refs.push(zone.intern_bytes(s.as_bytes())?);
            }
        }
        return Ok(PropertyValue::ArrStringRef(refs));
    }
    // Heterogeneous list → bincode fallback (send the entire
    // `Value::List` so the decoder round-trips the structure).
    encode_bincode_fallback(zone, original)
}

fn encode_bincode_fallback(zone: &mut PropsZone, v: &Value) -> SubstrateResult<PropertyValue> {
    let cfg = bincode_config();
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    buf.push(BYTES_DISCRIMINATOR_BINCODE);
    bincode::serde::encode_into_std_write(v, &mut buf, cfg)
        .map_err(|e| SubstrateError::WalBadFrame(format!("bincode encode failed: {e}")))?;
    let href = zone.intern_bytes(&buf)?;
    Ok(PropertyValue::BytesRef(href))
}

/// Convert a [`PropertyValue`] back into a [`Value`], dereferencing any
/// heap refs through `zone`.
pub fn decode_value(zone: &PropsZone, pv: &PropertyValue) -> SubstrateResult<Value> {
    match pv {
        PropertyValue::Null => Ok(Value::Null),
        PropertyValue::Bool(b) => Ok(Value::Bool(*b)),
        PropertyValue::I64(i) => Ok(Value::Int64(*i)),
        PropertyValue::F64(f) => Ok(Value::Float64(*f)),
        PropertyValue::StringRef(h) => decode_string_ref(zone, *h),
        PropertyValue::BytesRef(h) => decode_bytes_ref(zone, *h),
        PropertyValue::ArrI64(xs) => {
            let list: Vec<Value> = xs.iter().copied().map(Value::Int64).collect();
            Ok(Value::List(Arc::from(list.into_boxed_slice())))
        }
        PropertyValue::ArrF64(xs) => {
            let list: Vec<Value> = xs.iter().copied().map(Value::Float64).collect();
            Ok(Value::List(Arc::from(list.into_boxed_slice())))
        }
        PropertyValue::ArrStringRef(hs) => {
            let mut list = Vec::with_capacity(hs.len());
            for h in hs {
                match decode_string_ref(zone, *h)? {
                    s @ Value::String(_) => list.push(s),
                    other => {
                        return Err(SubstrateError::WalBadFrame(format!(
                            "ArrStringRef element decoded to non-String: {other:?}"
                        )))
                    }
                }
            }
            Ok(Value::List(Arc::from(list.into_boxed_slice())))
        }
    }
}

fn decode_string_ref(zone: &PropsZone, h: HeapRef) -> SubstrateResult<Value> {
    let bytes = zone.read_heap(h).ok_or_else(|| {
        SubstrateError::WalBadFrame(format!(
            "StringRef {{page_id={}, offset={}}} out of range",
            h.page_id, h.offset
        ))
    })?;
    let s = std::str::from_utf8(&bytes).map_err(|e| {
        SubstrateError::WalBadFrame(format!("StringRef payload not utf8: {e}"))
    })?;
    Ok(Value::String(ArcStr::from(s)))
}

fn decode_bytes_ref(zone: &PropsZone, h: HeapRef) -> SubstrateResult<Value> {
    let bytes = zone.read_heap(h).ok_or_else(|| {
        SubstrateError::WalBadFrame(format!(
            "BytesRef {{page_id={}, offset={}}} out of range",
            h.page_id, h.offset
        ))
    })?;
    if bytes.is_empty() {
        return Err(SubstrateError::WalBadFrame(
            "BytesRef payload missing discriminator byte".into(),
        ));
    }
    match bytes[0] {
        BYTES_DISCRIMINATOR_RAW => Ok(Value::Bytes(Arc::from(&bytes[1..]))),
        BYTES_DISCRIMINATOR_BINCODE => {
            let (decoded, _consumed) =
                bincode::serde::decode_from_slice::<Value, _>(&bytes[1..], bincode_config())
                    .map_err(|e| {
                        SubstrateError::WalBadFrame(format!(
                            "bincode decode of BytesRef payload failed: {e}"
                        ))
                    })?;
            Ok(decoded)
        }
        d => Err(SubstrateError::WalBadFrame(format!(
            "unknown BytesRef discriminator {d:#x}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::SubstrateFile;
    use obrain_common::types::Duration;
    use obrain_common::types::PropertyKey;
    use std::collections::BTreeMap;

    fn open_zone() -> (SubstrateFile, PropsZone) {
        let sf = SubstrateFile::open_tempfile().unwrap();
        let pz = PropsZone::open(&sf).unwrap();
        (sf, pz)
    }

    fn roundtrip(v: Value) -> Value {
        let (_sf, mut pz) = open_zone();
        let pv = encode_value(&mut pz, &v).expect("encode");
        decode_value(&pz, &pv).expect("decode")
    }

    #[test]
    fn null_roundtrips() {
        assert_eq!(roundtrip(Value::Null), Value::Null);
    }

    #[test]
    fn bool_roundtrips() {
        assert_eq!(roundtrip(Value::Bool(true)), Value::Bool(true));
        assert_eq!(roundtrip(Value::Bool(false)), Value::Bool(false));
    }

    #[test]
    fn int64_roundtrips() {
        for i in [i64::MIN, -42, 0, 1, 42, i64::MAX] {
            assert_eq!(roundtrip(Value::Int64(i)), Value::Int64(i));
        }
    }

    #[test]
    fn float64_roundtrips() {
        for f in [0.0, -1.5, core::f64::consts::PI, f64::MIN, f64::MAX] {
            assert_eq!(roundtrip(Value::Float64(f)), Value::Float64(f));
        }
    }

    #[test]
    fn float_nan_encodes_but_decode_is_bitwise_distinct() {
        // NaN != NaN under `==`, so we compare bit patterns.
        let (_sf, mut pz) = open_zone();
        let pv = encode_value(&mut pz, &Value::Float64(f64::NAN)).unwrap();
        let back = decode_value(&pz, &pv).unwrap();
        match back {
            Value::Float64(f) => assert!(f.is_nan()),
            other => panic!("expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn string_roundtrips() {
        let cases = ["", "a", "hello world", "unicode ✓ café 🚀", &"x".repeat(512)];
        for s in cases {
            let got = roundtrip(Value::String(ArcStr::from(s)));
            assert_eq!(got, Value::String(ArcStr::from(s)));
        }
    }

    #[test]
    fn bytes_roundtrip_preserves_every_byte_including_discriminator() {
        // A raw bytes payload that happens to start with 0x01 (the bincode
        // discriminator) — our encoder prepends 0x00 before interning, so
        // the decoder must strip exactly one byte.
        let bs: Arc<[u8]> = Arc::from([0x01, 0x02, 0x03].as_slice());
        let got = roundtrip(Value::Bytes(bs.clone()));
        assert_eq!(got, Value::Bytes(bs));
    }

    #[test]
    fn bytes_empty_roundtrips() {
        let empty: Arc<[u8]> = Arc::from([].as_slice());
        let got = roundtrip(Value::Bytes(empty.clone()));
        assert_eq!(got, Value::Bytes(empty));
    }

    #[test]
    fn homogeneous_int_list_takes_fast_path() {
        let (_sf, mut pz) = open_zone();
        let xs: Arc<[Value]> = Arc::from(
            [Value::Int64(1), Value::Int64(2), Value::Int64(3)].as_slice(),
        );
        let pv = encode_value(&mut pz, &Value::List(xs.clone())).unwrap();
        // Fast path: should be ArrI64, not BytesRef.
        assert!(matches!(pv, PropertyValue::ArrI64(ref v) if v == &vec![1i64, 2, 3]));
        let back = decode_value(&pz, &pv).unwrap();
        assert_eq!(back, Value::List(xs));
    }

    #[test]
    fn homogeneous_float_list_takes_fast_path() {
        let (_sf, mut pz) = open_zone();
        let xs: Arc<[Value]> = Arc::from(
            [Value::Float64(1.5), Value::Float64(-2.5), Value::Float64(0.0)].as_slice(),
        );
        let pv = encode_value(&mut pz, &Value::List(xs.clone())).unwrap();
        assert!(matches!(pv, PropertyValue::ArrF64(_)));
        let back = decode_value(&pz, &pv).unwrap();
        assert_eq!(back, Value::List(xs));
    }

    #[test]
    fn homogeneous_string_list_takes_fast_path() {
        let (_sf, mut pz) = open_zone();
        let xs: Arc<[Value]> = Arc::from(
            [
                Value::String(ArcStr::from("a")),
                Value::String(ArcStr::from("hello")),
                Value::String(ArcStr::from("")),
            ]
            .as_slice(),
        );
        let pv = encode_value(&mut pz, &Value::List(xs.clone())).unwrap();
        assert!(matches!(pv, PropertyValue::ArrStringRef(_)));
        let back = decode_value(&pz, &pv).unwrap();
        assert_eq!(back, Value::List(xs));
    }

    #[test]
    fn heterogeneous_list_falls_back_to_bincode() {
        let (_sf, mut pz) = open_zone();
        let xs: Arc<[Value]> = Arc::from(
            [
                Value::Int64(42),
                Value::String(ArcStr::from("mixed")),
                Value::Bool(true),
            ]
            .as_slice(),
        );
        let pv = encode_value(&mut pz, &Value::List(xs.clone())).unwrap();
        // Heterogeneous → BytesRef fallback.
        assert!(matches!(pv, PropertyValue::BytesRef(_)));
        let back = decode_value(&pz, &pv).unwrap();
        assert_eq!(back, Value::List(xs));
    }

    #[test]
    fn empty_list_roundtrips_via_arr_i64_canonical() {
        let (_sf, mut pz) = open_zone();
        let xs: Arc<[Value]> = Arc::from([].as_slice());
        let pv = encode_value(&mut pz, &Value::List(xs.clone())).unwrap();
        assert!(matches!(pv, PropertyValue::ArrI64(ref v) if v.is_empty()));
        let back = decode_value(&pz, &pv).unwrap();
        assert_eq!(back, Value::List(xs));
    }

    #[test]
    fn map_falls_back_to_bincode() {
        let mut m = BTreeMap::new();
        m.insert(PropertyKey::new("a"), Value::Int64(1));
        m.insert(PropertyKey::new("b"), Value::String(ArcStr::from("hello")));
        let v = Value::Map(Arc::new(m));
        let back = roundtrip(v.clone());
        assert_eq!(back, v);
    }

    #[test]
    fn duration_falls_back_to_bincode() {
        let v = Value::Duration(Duration::new(1, 2, 3_000_000_000));
        let back = roundtrip(v.clone());
        assert_eq!(back, v);
    }

    #[test]
    fn vector_is_rejected() {
        let (_sf, mut pz) = open_zone();
        let vec: Arc<[f32]> = Arc::from([0.1f32, 0.2, 0.3].as_slice());
        let err = encode_value(&mut pz, &Value::Vector(vec)).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn bad_discriminator_is_rejected() {
        // Force a BytesRef with an invalid discriminator into the heap
        // and verify decode fails cleanly.
        let (_sf, mut pz) = open_zone();
        let href = pz.intern_bytes(&[0xFE, 0xAA, 0xBB]).unwrap();
        let pv = PropertyValue::BytesRef(href);
        let err = decode_value(&pz, &pv).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn invalid_utf8_string_ref_is_rejected() {
        let (_sf, mut pz) = open_zone();
        let href = pz.intern_bytes(&[0xFF, 0xFE, 0xFD]).unwrap();
        let pv = PropertyValue::StringRef(href);
        let err = decode_value(&pz, &pv).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn dangling_heap_ref_is_rejected_cleanly() {
        // HeapRef pointing to a page that does not exist.
        let (_sf, pz) = open_zone();
        let bogus = HeapRef {
            page_id: 99_999,
            offset: 0,
        };
        let pv = PropertyValue::BytesRef(bogus);
        let err = decode_value(&pz, &pv).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));

        let pv2 = PropertyValue::StringRef(bogus);
        let err = decode_value(&pz, &pv2).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn many_values_end_to_end() {
        // Bulk encode a mix of every supported variant; verify each
        // decodes byte-identical.
        let (_sf, mut pz) = open_zone();
        let mut expected: Vec<Value> = Vec::new();
        let mut pvs: Vec<PropertyValue> = Vec::new();
        for i in 0..200i64 {
            let vs = [
                Value::Null,
                Value::Bool(i % 2 == 0),
                Value::Int64(i),
                Value::Float64(i as f64 * 0.5),
                Value::String(ArcStr::from(format!("str-{i}"))),
                Value::Bytes(Arc::from(vec![(i & 0xFF) as u8; 4].into_boxed_slice())),
                Value::List(Arc::from(
                    vec![Value::Int64(i), Value::Int64(i + 1)].into_boxed_slice(),
                )),
            ];
            for v in vs {
                pvs.push(encode_value(&mut pz, &v).unwrap());
                expected.push(v);
            }
        }
        for (pv, want) in pvs.iter().zip(expected.iter()) {
            let got = decode_value(&pz, pv).unwrap();
            assert_eq!(&got, want);
        }
    }
}

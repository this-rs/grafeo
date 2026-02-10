//! Converts between JavaScript and Grafeo value types.
//!
//! | JavaScript type  | Grafeo type   | Notes                          |
//! | ---------------- | ------------- | ------------------------------ |
//! | `null/undefined` | `Null`        |                                |
//! | `boolean`        | `Bool`        |                                |
//! | `number`         | `Int64/Float64` | Integer if no fractional part |
//! | `string`         | `String`      |                                |
//! | `Array`          | `List`        | Elements converted recursively |
//! | `Object`         | `Map`         | Keys must be strings           |
//! | `Buffer`         | `Bytes`       |                                |
//! | `Date`           | `Timestamp`   | Millisecond precision          |
//! | `BigInt`         | `Int64`       |                                |
//! | `Float32Array`   | `Vector`      |                                |

use std::collections::BTreeMap;
use std::ffi::CString;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::{JsDate, JsString, JsValue, ValueType, sys};

use grafeo_common::types::{PropertyKey, Timestamp, Value};

/// Converts a JavaScript value to a Grafeo Value.
pub fn js_to_value(env: &Env, val: Unknown<'_>) -> Result<Value> {
    #![allow(clippy::trivially_copy_pass_by_ref)] // Env refs are conventional in napi
    let value_type = val.get_type()?;
    match value_type {
        ValueType::Null | ValueType::Undefined => Ok(Value::Null),
        ValueType::Boolean => {
            let b = val.coerce_to_bool()?;
            Ok(Value::Bool(b))
        }
        ValueType::Number => {
            let n: f64 = val.coerce_to_number()?.get_double()?;
            // If the number is an integer within safe range, store as Int64
            if n.fract() == 0.0 && n.abs() < (1i64 << 53) as f64 {
                Ok(Value::Int64(n as i64))
            } else {
                Ok(Value::Float64(n))
            }
        }
        ValueType::String => {
            let s = val.coerce_to_string()?.into_utf8()?.into_owned()?;
            Ok(Value::String(s.into()))
        }
        ValueType::BigInt => {
            let bigint: BigInt = unsafe { val.cast()? };
            let word = if bigint.words.is_empty() {
                0u64
            } else {
                bigint.words[0]
            };
            let signed = if bigint.sign_bit {
                -(word as i64)
            } else {
                word as i64
            };
            Ok(Value::Int64(signed))
        }
        ValueType::Object => {
            let obj: Object<'_> = unsafe { val.cast()? };
            js_object_to_value(env, &obj)
        }
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Unsupported JavaScript type: {:?}", value_type),
        )),
    }
}

/// Converts a JavaScript object (Array, Buffer, Date, or plain object) to a Grafeo Value.
fn js_object_to_value(env: &Env, obj: &Object<'_>) -> Result<Value> {
    if obj.is_array()? {
        let len = obj.get_array_length()?;
        let mut items = Vec::with_capacity(len as usize);
        for i in 0..len {
            let elem: Unknown<'_> = obj.get_element(i)?;
            items.push(js_to_value(env, elem)?);
        }
        return Ok(Value::List(items.into()));
    }

    if obj.is_buffer()? {
        let unknown = unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()) };
        let buf: Buffer = unsafe { unknown.cast()? };
        return Ok(Value::Bytes(buf.to_vec().into()));
    }

    if obj.is_date()? {
        let date: JsDate = unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
        let ms = date.value_of()?;
        let micros = (ms * 1000.0) as i64;
        return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
    }

    // Check for TypedArray (Float32Array for vectors)
    if obj.is_typedarray()? {
        let ta: TypedArray<'_> =
            unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
        if ta.typed_array_type == TypedArrayType::Float32 {
            let f32arr: Float32Array =
                unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
            return Ok(Value::Vector(f32arr.to_vec().into()));
        }
    }

    // Plain object -> Map
    let keys = obj.get_property_names()?;
    let len = keys.get_array_length()?;
    let mut map = BTreeMap::new();
    for i in 0..len {
        let key: JsString = keys.get_element(i)?;
        let key_str = key.into_utf8()?.into_owned()?;
        let value: Unknown<'_> = obj.get_named_property(&key_str)?;
        map.insert(PropertyKey::new(key_str), js_to_value(env, value)?);
    }
    Ok(Value::Map(Arc::new(map)))
}

/// Helper to check napi_status and convert to Result.
pub(crate) fn check_napi(status: sys::napi_status) -> Result<()> {
    if status == sys::Status::napi_ok {
        Ok(())
    } else {
        Err(napi::Error::new(
            napi::Status::GenericFailure,
            format!("napi call failed with status: {status:?}"),
        ))
    }
}

/// Converts a Grafeo Value to a raw napi value (no lifetime constraints).
///
/// This uses the raw napi C API to avoid lifetime issues when returning
/// JS values from `#[napi]` methods where `env` is taken by value.
pub fn value_to_napi(env: sys::napi_env, value: &Value) -> Result<sys::napi_value> {
    match value {
        Value::Null => unsafe { <Null as ToNapiValue>::to_napi_value(env, Null) },
        Value::Bool(b) => unsafe { <bool as ToNapiValue>::to_napi_value(env, *b) },
        Value::Int64(i) => {
            // Use number for safe integer range, BigInt for larger values
            if *i > -(1i64 << 53) && *i < (1i64 << 53) {
                unsafe { <i64 as ToNapiValue>::to_napi_value(env, *i) }
            } else {
                unsafe {
                    <BigInt as ToNapiValue>::to_napi_value(
                        env,
                        BigInt {
                            sign_bit: *i < 0,
                            words: vec![i.unsigned_abs()],
                        },
                    )
                }
            }
        }
        Value::Float64(f) => unsafe { <f64 as ToNapiValue>::to_napi_value(env, *f) },
        Value::String(s) => unsafe { <&str as ToNapiValue>::to_napi_value(env, s.as_ref()) },
        Value::List(items) => {
            let mut arr = std::ptr::null_mut();
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, items.len(), &raw mut arr)
            })?;
            for (i, item) in items.iter().enumerate() {
                let val = value_to_napi(env, item)?;
                check_napi(unsafe { sys::napi_set_element(env, arr, i as u32, val) })?;
            }
            Ok(arr)
        }
        Value::Map(map) => {
            let mut obj = std::ptr::null_mut();
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            for (key, val) in map.as_ref() {
                let key_cstr = CString::new(key.as_str())
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
                let napi_val = value_to_napi(env, val)?;
                check_napi(unsafe {
                    sys::napi_set_named_property(env, obj, key_cstr.as_ptr(), napi_val)
                })?;
            }
            Ok(obj)
        }
        Value::Bytes(bytes) => unsafe {
            <Buffer as ToNapiValue>::to_napi_value(env, Buffer::from(bytes.to_vec()))
        },
        Value::Timestamp(ts) => {
            let ms = ts.as_micros() as f64 / 1000.0;
            let env_wrapper = Env::from_raw(env);
            Ok(env_wrapper.create_date(ms)?.raw())
        }
        Value::Vector(v) => unsafe {
            <Float32Array as ToNapiValue>::to_napi_value(env, Float32Array::new(v.to_vec()))
        },
    }
}

/// Converts a Grafeo Value to a JavaScript Unknown value.
///
/// Uses `value_to_napi` internally and wraps the result as `Unknown`.
/// The lifetime is unconstrained (from `from_raw_unchecked`), so this is
/// safe to call from `#[napi]` methods where `env` is taken by value.
pub fn value_to_js(env: sys::napi_env, value: &Value) -> Result<Unknown<'_>> {
    let raw = value_to_napi(env, value)?;
    Ok(unsafe { Unknown::from_raw_unchecked(env, raw) })
}

//! LOAD CSV operator for reading CSV files and producing rows.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use grafeo_common::types::{ArcStr, LogicalType, PropertyKey, Value};

/// Operator that reads a CSV file and produces one row per CSV record.
///
/// With headers: each row is a `Value::Map` with column names as keys.
/// Without headers: each row is a `Value::List` of string values.
pub struct LoadCsvOperator {
    /// Buffered reader for the CSV file.
    reader: Option<BufReader<File>>,
    /// Column headers (if WITH HEADERS).
    headers: Option<Vec<String>>,
    /// Whether the CSV has headers.
    with_headers: bool,
    /// File path (for reset).
    path: String,
    /// Field separator.
    delimiter: u8,
    /// Whether the file has been opened.
    opened: bool,
}

impl LoadCsvOperator {
    /// Creates a new LOAD CSV operator.
    pub fn new(
        path: String,
        with_headers: bool,
        field_terminator: Option<char>,
        _variable: String,
    ) -> Self {
        let delimiter = field_terminator.map_or(b',', |c| {
            let mut buf = [0u8; 4];
            c.encode_utf8(&mut buf);
            buf[0]
        });

        Self {
            reader: None,
            headers: None,
            with_headers,
            path,
            delimiter,
            opened: false,
        }
    }

    /// Opens the file and reads headers if needed.
    fn open(&mut self) -> Result<(), OperatorError> {
        // Strip file:/// prefix if present (Neo4j convention)
        let file_path = self
            .path
            .strip_prefix("file:///")
            .or_else(|| self.path.strip_prefix("file://"))
            .unwrap_or(&self.path);

        let file = File::open(file_path).map_err(|e| {
            OperatorError::Execution(format!("Failed to open CSV file '{}': {}", self.path, e))
        })?;
        let mut reader = BufReader::new(file);

        if self.with_headers {
            let mut header_line = String::new();
            reader.read_line(&mut header_line).map_err(|e| {
                OperatorError::Execution(format!("Failed to read CSV headers: {e}"))
            })?;
            // Strip BOM if present
            let header_line = header_line.strip_prefix('\u{feff}').unwrap_or(&header_line);
            let header_line = header_line.trim_end_matches(['\r', '\n']);
            self.headers = Some(parse_csv_row(header_line, self.delimiter));
        }

        self.reader = Some(reader);
        self.opened = true;
        Ok(())
    }
}

impl Operator for LoadCsvOperator {
    fn next(&mut self) -> OperatorResult {
        if !self.opened {
            self.open()?;
        }

        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| OperatorError::Execution("CSV reader not initialized".to_string()))?;

        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .map_err(|e| OperatorError::Execution(format!("Failed to read CSV line: {e}")))?;

            if bytes_read == 0 {
                return Ok(None); // EOF
            }

            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                continue; // skip blank lines
            }

            let fields = parse_csv_row(trimmed, self.delimiter);

            let row_value = if let Some(headers) = &self.headers {
                // WITH HEADERS: produce a Map
                let mut map = BTreeMap::new();
                for (i, header) in headers.iter().enumerate() {
                    let value = fields.get(i).map_or(Value::Null, |s| {
                        if s.is_empty() {
                            Value::Null
                        } else {
                            Value::String(ArcStr::from(s.as_str()))
                        }
                    });
                    map.insert(PropertyKey::from(header.as_str()), value);
                }
                Value::Map(Arc::new(map))
            } else {
                // Without headers: produce a List
                let values: Vec<Value> = fields
                    .into_iter()
                    .map(|s| {
                        if s.is_empty() {
                            Value::Null
                        } else {
                            Value::String(ArcStr::from(s.as_str()))
                        }
                    })
                    .collect();
                Value::List(Arc::from(values))
            };

            // Build a single-row DataChunk with one column (the row variable)
            let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
            if let Some(col) = builder.column_mut(0) {
                col.push_value(row_value);
            }
            builder.advance_row();
            return Ok(Some(builder.finish()));
        }
    }

    fn reset(&mut self) {
        self.reader = None;
        self.headers = None;
        self.opened = false;
    }

    fn name(&self) -> &'static str {
        "LoadCsv"
    }
}

/// Parses a single CSV row into fields, respecting quoted fields.
///
/// Handles:
/// - Unquoted fields separated by the delimiter
/// - Double-quoted fields (can contain delimiters, newlines, and escaped quotes)
/// - Escaped quotes within quoted fields (`""` becomes `"`)
fn parse_csv_row(line: &str, delimiter: u8) -> Vec<String> {
    let delim = delimiter as char;
    let mut fields = Vec::new();
    let mut chars = line.chars().peekable();
    let mut field = String::new();

    loop {
        if chars.peek() == Some(&'"') {
            // Quoted field
            chars.next(); // consume opening quote
            loop {
                match chars.next() {
                    Some('"') => {
                        if chars.peek() == Some(&'"') {
                            // Escaped quote
                            chars.next();
                            field.push('"');
                        } else {
                            // End of quoted field
                            break;
                        }
                    }
                    Some(c) => field.push(c),
                    None => break, // Unterminated quote, take what we have
                }
            }
            // Skip to delimiter or end
            match chars.peek() {
                Some(c) if *c == delim => {
                    chars.next();
                }
                _ => {}
            }
            fields.push(std::mem::take(&mut field));
        } else {
            // Unquoted field
            loop {
                match chars.peek() {
                    Some(c) if *c == delim => {
                        chars.next();
                        break;
                    }
                    Some(_) => {
                        field.push(chars.next().unwrap());
                    }
                    None => break,
                }
            }
            fields.push(std::mem::take(&mut field));
        }

        if chars.peek().is_none() {
            break;
        }
    }

    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_simple() {
        let fields = parse_csv_row("a,b,c", b',');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_csv_quoted() {
        let fields = parse_csv_row(r#""hello","world""#, b',');
        assert_eq!(fields, vec!["hello", "world"]);
    }

    #[test]
    fn test_parse_csv_escaped_quotes() {
        let fields = parse_csv_row(r#""say ""hi""","ok""#, b',');
        assert_eq!(fields, vec![r#"say "hi""#, "ok"]);
    }

    #[test]
    fn test_parse_csv_delimiter_in_quoted() {
        let fields = parse_csv_row(r#""a,b",c"#, b',');
        assert_eq!(fields, vec!["a,b", "c"]);
    }

    #[test]
    fn test_parse_csv_empty_fields() {
        let fields = parse_csv_row("a,,c", b',');
        assert_eq!(fields, vec!["a", "", "c"]);
    }

    #[test]
    fn test_parse_csv_tab_delimiter() {
        let fields = parse_csv_row("a\tb\tc", b'\t');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_csv_single_field() {
        let fields = parse_csv_row("hello", b',');
        assert_eq!(fields, vec!["hello"]);
    }
}

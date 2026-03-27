//! Data export/import commands.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

use anyhow::{Context, Result};

use grafeo_common::types::Value;

use crate::output;
use crate::{DataCommands, OutputFormat};

/// Run data commands.
pub fn run(cmd: DataCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        DataCommands::Dump {
            path,
            output: out,
            export_format,
        } => {
            let format_name = export_format.as_deref().unwrap_or("json");
            if format_name != "json" && format_name != "jsonl" {
                anyhow::bail!(
                    "Only JSON Lines format is currently supported.\n\
                     Use `--export-format json` or omit for the default."
                );
            }

            let db = super::open_existing(&path)?;
            let file = std::fs::File::create(&out)
                .with_context(|| format!("Failed to create output file: {}", out.display()))?;
            let mut writer = std::io::BufWriter::new(file);

            let mut node_count = 0usize;
            for node in db.iter_nodes() {
                let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
                let properties: HashMap<String, Value> = node
                    .properties
                    .to_btree_map()
                    .into_iter()
                    .map(|(k, v)| (k.as_str().to_string(), v))
                    .collect();
                let record = serde_json::json!({
                    "type": "node",
                    "id": node.id.0,
                    "labels": labels,
                    "properties": properties,
                });
                serde_json::to_writer(&mut writer, &record)?;
                writeln!(writer)?;
                node_count += 1;
            }

            let mut edge_count = 0usize;
            for edge in db.iter_edges() {
                let properties: HashMap<String, Value> = edge
                    .properties
                    .to_btree_map()
                    .into_iter()
                    .map(|(k, v)| (k.as_str().to_string(), v))
                    .collect();
                let record = serde_json::json!({
                    "type": "edge",
                    "id": edge.id.0,
                    "source": edge.src.0,
                    "target": edge.dst.0,
                    "edge_type": edge.edge_type.as_str(),
                    "properties": properties,
                });
                serde_json::to_writer(&mut writer, &record)?;
                writeln!(writer)?;
                edge_count += 1;
            }

            writer.flush()?;
            output::status(
                &format!(
                    "Exported {} nodes and {} edges to {}",
                    node_count,
                    edge_count,
                    out.display()
                ),
                quiet,
            );
        }
        DataCommands::Load { input, path } => {
            let file = std::fs::File::open(&input)
                .with_context(|| format!("Failed to open input file: {}", input.display()))?;
            let reader = BufReader::new(file);

            let db = if path.exists() {
                super::open_existing(&path)?
            } else {
                grafeo_engine::GrafeoDB::open(&path)
                    .with_context(|| format!("Failed to create database at {}", path.display()))?
            };

            let mut node_count = 0usize;
            let mut edge_count = 0usize;

            for (line_num, line) in reader.lines().enumerate() {
                let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let record: serde_json::Value = serde_json::from_str(trimmed)
                    .with_context(|| format!("Invalid JSON on line {}", line_num + 1))?;

                match record.get("type").and_then(|t| t.as_str()) {
                    Some("node") => {
                        let labels: Vec<&str> = record
                            .get("labels")
                            .and_then(|l| l.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                            .unwrap_or_default();

                        let id = db.create_node(&labels);

                        if let Some(props) = record.get("properties").and_then(|p| p.as_object()) {
                            for (key, val) in props {
                                let value: Value =
                                    serde_json::from_value(val.clone()).unwrap_or(Value::Null);
                                db.set_node_property(id, key, value);
                            }
                        }
                        node_count += 1;
                    }
                    Some("edge") => {
                        let source = record
                            .get("source")
                            .and_then(|v| v.as_u64())
                            .map(grafeo_common::types::NodeId)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing 'source' on line {}", line_num + 1)
                            })?;
                        let target = record
                            .get("target")
                            .and_then(|v| v.as_u64())
                            .map(grafeo_common::types::NodeId)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing 'target' on line {}", line_num + 1)
                            })?;
                        let edge_type = record
                            .get("edge_type")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing 'edge_type' on line {}", line_num + 1)
                            })?;

                        let id = db.create_edge(source, target, edge_type);

                        if let Some(props) = record.get("properties").and_then(|p| p.as_object()) {
                            for (key, val) in props {
                                let value: Value =
                                    serde_json::from_value(val.clone()).unwrap_or(Value::Null);
                                db.set_edge_property(id, key, value);
                            }
                        }
                        edge_count += 1;
                    }
                    other => {
                        anyhow::bail!("Unknown record type {:?} on line {}", other, line_num + 1);
                    }
                }
            }

            output::status(
                &format!(
                    "Loaded {} nodes and {} edges into {}",
                    node_count,
                    edge_count,
                    path.display()
                ),
                quiet,
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &std::path::Path) -> grafeo_engine::GrafeoDB {
        let db = grafeo_engine::GrafeoDB::open(dir).expect("create db");
        let n1 = db.create_node(&["Person"]);
        let n2 = db.create_node(&["Company"]);
        db.set_node_property(n1, "name", Value::from("Alix"));
        db.set_node_property(n1, "age", Value::Int64(30));
        db.set_node_property(n2, "name", Value::from("Acme"));
        let e = db.create_edge(n1, n2, "WORKS_AT");
        db.set_edge_property(e, "since", Value::Int64(2020));
        db
    }

    #[test]
    fn test_dump_and_load_roundtrip() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("source.grafeo");
        let dump_path = temp.path().join("dump.jsonl");
        let target_path = temp.path().join("target.grafeo");

        let db = create_test_db(&db_path);
        drop(db);

        // Dump
        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true, // quiet
        )
        .expect("dump should succeed");

        // Verify dump file exists and has content
        let content = std::fs::read_to_string(&dump_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3); // 2 nodes + 1 edge

        // Load into new database
        run(
            DataCommands::Load {
                input: dump_path,
                path: target_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load should succeed");

        // Verify loaded data
        let loaded = grafeo_engine::GrafeoDB::open(&target_path).unwrap();
        let info = loaded.info();
        assert_eq!(info.node_count, 2);
        assert_eq!(info.edge_count, 1);
    }

    #[test]
    fn test_dump_explicit_json_format() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.json");

        // Create and drop the database so the CLI can reopen it
        drop(create_test_db(&db_path));

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: Some("json".to_string()),
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump with explicit json format should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        assert!(!content.is_empty());
    }

    #[test]
    fn test_dump_invalid_format_rejected() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.parquet");

        let _db = create_test_db(&db_path);

        let result = run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path,
                export_format: Some("parquet".to_string()),
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("JSON Lines"));
    }

    #[test]
    fn test_load_invalid_json_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("bad.jsonl");
        let db_path = temp.path().join("target.grafeo");

        std::fs::write(&input_path, "not valid json\n").unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("line 1"));
    }

    #[test]
    fn test_load_unknown_type_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("unknown.jsonl");
        let db_path = temp.path().join("target.grafeo");

        std::fs::write(&input_path, "{\"type\": \"widget\"}\n").unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unknown record type"));
    }

    #[test]
    fn test_load_skips_empty_lines() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("sparse.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "\n{\"type\":\"node\",\"labels\":[\"A\"],\"properties\":{}}\n\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("should handle empty lines");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_load_edge_missing_source_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("bad_edge.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "{\"type\":\"edge\",\"target\":1,\"edge_type\":\"KNOWS\"}\n";
        std::fs::write(&input_path, content).unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("source"));
    }
}

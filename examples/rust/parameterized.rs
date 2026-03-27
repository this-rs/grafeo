//! Parameterized queries: safe, reusable query execution.
//!
//! Run with: `cargo run -p obrain-examples --bin parameterized`

use std::collections::HashMap;

use obrain::{ObrainDB, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    // ── Insert products using parameterized queries ────────────────
    // Parameters use $name syntax in the query and are passed as a
    // HashMap<String, Value>. This prevents injection and lets you
    // reuse the same query template with different values.

    let insert_query = "INSERT (:Product {name: $name, price: $price, category: $category})";

    // Each call reuses the same query template with different parameters
    let products = [
        ("Espresso Machine", 299.99, "Kitchen"),
        ("Mechanical Keyboard", 149.95, "Electronics"),
        ("Desk Lamp", 45.50, "Office"),
        ("Standing Desk", 599.00, "Office"),
        ("Monitor", 429.99, "Electronics"),
        ("Coffee Grinder", 89.95, "Kitchen"),
    ];

    for (name, price, category) in products {
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::from(name));
        params.insert("price".to_string(), Value::from(price));
        params.insert("category".to_string(), Value::from(category));

        session.execute_with_params(insert_query, params)?;
    }

    println!("Inserted {} products\n", products.len());

    // ── Query with a single string parameter ──────────────────────
    let mut params = HashMap::new();
    params.insert("cat".to_string(), Value::from("Kitchen"));

    let result = session.execute_with_params(
        "MATCH (p:Product)
         WHERE p.category = $cat
         RETURN p.name, p.price
         ORDER BY p.price",
        params,
    )?;

    println!("Kitchen products:");
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let price = row[1].as_float64().unwrap_or(0.0);
        println!("  {:<25} ${:.2}", name, price);
    }

    // ── Query with numeric range parameters ───────────────────────
    // The same query template works with different price ranges.
    let range_query = "MATCH (p:Product)
         WHERE p.price >= $min_price AND p.price <= $max_price
         RETURN p.name, p.price, p.category
         ORDER BY p.price";

    // Range 1: budget products (under $100)
    let mut params = HashMap::new();
    params.insert("min_price".to_string(), Value::from(0.0));
    params.insert("max_price".to_string(), Value::from(100.0));

    let result = session.execute_with_params(range_query, params)?;
    println!("\nBudget products (under $100):");
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let price = row[1].as_float64().unwrap_or(0.0);
        let category = row[2].as_str().unwrap_or("?");
        println!("  {:<25} ${:<8.2} [{}]", name, price, category);
    }

    // Range 2: premium products ($200+)
    let mut params = HashMap::new();
    params.insert("min_price".to_string(), Value::from(200.0));
    params.insert("max_price".to_string(), Value::from(1000.0));

    let result = session.execute_with_params(range_query, params)?;
    println!("\nPremium products ($200+):");
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let price = row[1].as_float64().unwrap_or(0.0);
        let category = row[2].as_str().unwrap_or("?");
        println!("  {:<25} ${:<8.2} [{}]", name, price, category);
    }

    // ── Count with parameter ──────────────────────────────────────
    let mut params = HashMap::new();
    params.insert("cat".to_string(), Value::from("Electronics"));

    let count: i64 = session
        .execute_with_params(
            "MATCH (p:Product)
             WHERE p.category = $cat
             RETURN COUNT(p)",
            params,
        )?
        .scalar()?;

    println!("\nElectronics count: {count}");

    println!("\nDone!");
    Ok(())
}

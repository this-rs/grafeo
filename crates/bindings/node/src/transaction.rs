//! Transaction support for the Node.js API.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_engine::database::GrafeoDB;

use crate::error::NodeGrafeoError;
use crate::query::QueryResult;

/// A database transaction with explicit commit/rollback.
///
/// In Node.js 22+, use with `using` for automatic cleanup:
/// ```js
/// using tx = db.beginTransaction();
/// await tx.execute("INSERT (:Person {name: 'Alice'})");
/// tx.commit();
/// // auto-rollback if commit not called
/// ```
#[napi]
pub struct Transaction {
    db: Arc<RwLock<GrafeoDB>>,
    session: parking_lot::Mutex<Option<grafeo_engine::session::Session>>,
    committed: bool,
    rolled_back: bool,
}

#[napi]
impl Transaction {
    /// Execute a query within this transaction.
    #[napi]
    #[allow(clippy::unused_async)] // async required for napi Promise return
    pub async fn execute(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        if self.committed || self.rolled_back {
            return Err(
                NodeGrafeoError::Transaction("Transaction is no longer active".into()).into(),
            );
        }
        let session_guard = self.session.lock();
        let session = session_guard.as_ref().ok_or_else(|| {
            napi::Error::from(NodeGrafeoError::Transaction(
                "Transaction is no longer active".into(),
            ))
        })?;

        let param_map = grafeo_bindings_common::json::json_params_to_map(params.as_ref())
            .map_err(|msg| napi::Error::from(NodeGrafeoError::InvalidArgument(msg)))?;

        let result = session
            .execute_language(&query, "gql", param_map)
            .map_err(NodeGrafeoError::from)?;

        let db = self.db.read();
        let (nodes, edges) = crate::database::extract_entities(&result, &db);

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            nodes,
            edges,
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Commit the transaction.
    #[napi]
    pub fn commit(&mut self) -> Result<()> {
        if self.committed {
            return Err(NodeGrafeoError::Transaction("Already committed".into()).into());
        }
        if self.rolled_back {
            return Err(NodeGrafeoError::Transaction("Already rolled back".into()).into());
        }
        let mut session_guard = self.session.lock();
        if let Some(ref mut session) = *session_guard {
            session.commit().map_err(NodeGrafeoError::from)?;
        }
        self.committed = true;
        Ok(())
    }

    /// Roll back the transaction.
    #[napi]
    pub fn rollback(&mut self) -> Result<()> {
        if self.committed {
            return Err(NodeGrafeoError::Transaction("Already committed".into()).into());
        }
        if self.rolled_back {
            return Err(NodeGrafeoError::Transaction("Already rolled back".into()).into());
        }
        let mut session_guard = self.session.lock();
        if let Some(ref mut session) = *session_guard {
            session.rollback().map_err(NodeGrafeoError::from)?;
        }
        self.rolled_back = true;
        Ok(())
    }

    /// Whether the transaction is still active.
    #[napi(getter, js_name = "isActive")]
    pub fn is_active(&self) -> bool {
        !self.committed && !self.rolled_back
    }
}

impl Transaction {
    pub(crate) fn new(db: Arc<RwLock<GrafeoDB>>, isolation_level: Option<&str>) -> Result<Self> {
        // Parse isolation level string
        let level = match isolation_level {
            Some("read_committed") => {
                Some(grafeo_engine::transaction::IsolationLevel::ReadCommitted)
            }
            Some("serializable") => Some(grafeo_engine::transaction::IsolationLevel::Serializable),
            Some("snapshot") | None => None, // snapshot is the default
            Some(other) => {
                return Err(NodeGrafeoError::InvalidArgument(format!(
                    "Unknown isolation level '{}'. Use 'read_committed', 'snapshot', or 'serializable'",
                    other
                ))
                .into());
            }
        };

        let mut session = {
            let db_guard = db.read();
            db_guard.session()
        };

        // Begin the transaction with the specified isolation level
        if let Some(level) = level {
            session
                .begin_tx_with_isolation(level)
                .map_err(NodeGrafeoError::from)?;
        } else {
            session.begin_tx().map_err(NodeGrafeoError::from)?;
        }

        Ok(Self {
            db,
            session: parking_lot::Mutex::new(Some(session)),
            committed: false,
            rolled_back: false,
        })
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // Auto-rollback on drop if not explicitly committed or rolled back
        if !self.committed && !self.rolled_back {
            let mut session_guard = self.session.lock();
            if let Some(ref mut session) = *session_guard {
                let _ = session.rollback();
            }
        }
    }
}

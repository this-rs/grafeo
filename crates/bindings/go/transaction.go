package obrain

/*
#include "obrain.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Transaction represents a database transaction with explicit commit/rollback.
// If neither Commit nor Rollback is called, the transaction is automatically
// rolled back when garbage collected.
type Transaction struct {
	handle *C.ObrainTransaction
}

// BeginTransaction starts a new transaction with default isolation (snapshot).
func (db *Database) BeginTransaction() (*Transaction, error) {
	h := C.obrain_begin_transaction(db.handle)
	if h == nil {
		return nil, lastError()
	}
	tx := &Transaction{handle: h}
	runtime.SetFinalizer(tx, (*Transaction).free)
	return tx, nil
}

// BeginTransactionWith starts a transaction with a specific isolation level.
func (db *Database) BeginTransactionWith(level IsolationLevel) (*Transaction, error) {
	h := C.obrain_begin_transaction_with_isolation(db.handle, C.int32_t(level))
	if h == nil {
		return nil, lastError()
	}
	tx := &Transaction{handle: h}
	runtime.SetFinalizer(tx, (*Transaction).free)
	return tx, nil
}

// Execute runs a query within this transaction.
func (tx *Transaction) Execute(query string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	r := C.obrain_transaction_execute(tx.handle, cQuery)
	if r == nil {
		return nil, lastError()
	}
	defer C.obrain_free_result(r)
	return parseResult(r)
}

// ExecuteWithParams runs a query with JSON parameters within this transaction.
func (tx *Transaction) ExecuteWithParams(query string, paramsJSON string) (*QueryResult, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cParams := C.CString(paramsJSON)
	defer C.free(unsafe.Pointer(cParams))
	r := C.obrain_transaction_execute_with_params(tx.handle, cQuery, cParams)
	if r == nil {
		return nil, lastError()
	}
	defer C.obrain_free_result(r)
	return parseResult(r)
}

// Commit commits the transaction.
func (tx *Transaction) Commit() error {
	err := statusToError(C.obrain_commit(tx.handle))
	if err == nil {
		// Prevent double-free on GC.
		runtime.SetFinalizer(tx, nil)
	}
	return err
}

// Rollback aborts the transaction.
func (tx *Transaction) Rollback() error {
	err := statusToError(C.obrain_rollback(tx.handle))
	if err == nil {
		runtime.SetFinalizer(tx, nil)
	}
	return err
}

// free is the GC finalizer — auto-rollback + free if user forgot.
func (tx *Transaction) free() {
	if tx.handle != nil {
		C.obrain_free_transaction(tx.handle)
		tx.handle = nil
	}
}

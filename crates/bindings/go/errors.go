package obrain

/*
#include "obrain.h"
*/
import "C"
import (
	"errors"
	"fmt"
)

// ErrDatabase is the base error for all Obrain database errors.
var ErrDatabase = errors.New("obrain")

// lastError reads the thread-local error from the C layer.
func lastError() error {
	msg := C.obrain_last_error()
	if msg == nil {
		return fmt.Errorf("%w: unknown error", ErrDatabase)
	}
	return fmt.Errorf("%w: %s", ErrDatabase, C.GoString(msg))
}

// statusToError converts a ObrainStatus to a Go error (nil on success).
func statusToError(status C.ObrainStatus) error {
	if status == C.OBRAIN_OK {
		return nil
	}
	return lastError()
}

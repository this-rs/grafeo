/* Obrain C API
 *
 * Link against libobrain_c.so (Linux), libobrain_c.dylib (macOS),
 * or obrain_c.dll (Windows).
 *
 * Memory management:
 *   - Opaque pointers must be freed with their obrain_free_* function.
 *   - Strings documented as "free with obrain_free_string" are caller-owned.
 *   - Pointers documented as "valid until free" must NOT be freed separately.
 *
 * Error handling:
 *   - Functions return ObrainStatus (0 = success).
 *   - On error, call obrain_last_error() for a human-readable message.
 */

#ifndef OBRAIN_H
#define OBRAIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes -------------------------------------------------------- */

typedef enum {
    OBRAIN_OK                  = 0,
    OBRAIN_ERROR_DATABASE      = 1,
    OBRAIN_ERROR_QUERY         = 2,
    OBRAIN_ERROR_TRANSACTION   = 3,
    OBRAIN_ERROR_STORAGE       = 4,
    OBRAIN_ERROR_IO            = 5,
    OBRAIN_ERROR_SERIALIZATION = 6,
    OBRAIN_ERROR_INTERNAL      = 7,
    OBRAIN_ERROR_NULL_POINTER  = 8,
    OBRAIN_ERROR_INVALID_UTF8  = 9
} ObrainStatus;

/* ---- Opaque types -------------------------------------------------------- */

typedef struct ObrainDatabase    ObrainDatabase;
typedef struct ObrainTransaction ObrainTransaction;
typedef struct ObrainResult      ObrainResult;
typedef struct ObrainNode        ObrainNode;
typedef struct ObrainEdge        ObrainEdge;

/* ---- Error handling ------------------------------------------------------ */

const char* obrain_last_error(void);
void        obrain_clear_error(void);

/* ---- Lifecycle ----------------------------------------------------------- */

ObrainDatabase* obrain_open_memory(void);
ObrainDatabase* obrain_open(const char* path);
ObrainStatus    obrain_close(ObrainDatabase* db);
void            obrain_free_database(ObrainDatabase* db);
const char*     obrain_version(void);

/* ---- Query execution ----------------------------------------------------- */

ObrainResult* obrain_execute(ObrainDatabase* db, const char* query);
ObrainResult* obrain_execute_with_params(ObrainDatabase* db, const char* query, const char* params_json);
ObrainResult* obrain_execute_cypher(ObrainDatabase* db, const char* query);
ObrainResult* obrain_execute_gremlin(ObrainDatabase* db, const char* query);
ObrainResult* obrain_execute_graphql(ObrainDatabase* db, const char* query);
ObrainResult* obrain_execute_sparql(ObrainDatabase* db, const char* query);
ObrainResult* obrain_execute_sql(ObrainDatabase* db, const char* query);

/* ---- Result access ------------------------------------------------------- */

const char* obrain_result_json(const ObrainResult* result);
size_t      obrain_result_row_count(const ObrainResult* result);
double      obrain_result_execution_time_ms(const ObrainResult* result);
uint64_t    obrain_result_rows_scanned(const ObrainResult* result);
void        obrain_free_result(ObrainResult* result);

/* ---- Node CRUD ----------------------------------------------------------- */

uint64_t     obrain_create_node(ObrainDatabase* db, const char* labels_json, const char* properties_json);
ObrainStatus obrain_get_node(ObrainDatabase* db, uint64_t id, ObrainNode** out);
int32_t      obrain_delete_node(ObrainDatabase* db, uint64_t id);
ObrainStatus obrain_set_node_property(ObrainDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      obrain_remove_node_property(ObrainDatabase* db, uint64_t id, const char* key);
int32_t      obrain_add_node_label(ObrainDatabase* db, uint64_t id, const char* label);
int32_t      obrain_remove_node_label(ObrainDatabase* db, uint64_t id, const char* label);
char*        obrain_get_node_labels(ObrainDatabase* db, uint64_t id);

uint64_t    obrain_node_id(const ObrainNode* node);
const char* obrain_node_labels_json(const ObrainNode* node);
const char* obrain_node_properties_json(const ObrainNode* node);
void        obrain_free_node(ObrainNode* node);

/* ---- Edge CRUD ----------------------------------------------------------- */

uint64_t     obrain_create_edge(ObrainDatabase* db, uint64_t source_id, uint64_t target_id, const char* edge_type, const char* properties_json);
ObrainStatus obrain_get_edge(ObrainDatabase* db, uint64_t id, ObrainEdge** out);
int32_t      obrain_delete_edge(ObrainDatabase* db, uint64_t id);
ObrainStatus obrain_set_edge_property(ObrainDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      obrain_remove_edge_property(ObrainDatabase* db, uint64_t id, const char* key);

uint64_t    obrain_edge_id(const ObrainEdge* edge);
uint64_t    obrain_edge_source_id(const ObrainEdge* edge);
uint64_t    obrain_edge_target_id(const ObrainEdge* edge);
const char* obrain_edge_type(const ObrainEdge* edge);
const char* obrain_edge_properties_json(const ObrainEdge* edge);
void        obrain_free_edge(ObrainEdge* edge);

/* ---- Property indexes ---------------------------------------------------- */

ObrainStatus obrain_create_property_index(ObrainDatabase* db, const char* property);
int32_t      obrain_drop_property_index(ObrainDatabase* db, const char* property);
int32_t      obrain_has_property_index(ObrainDatabase* db, const char* property);
ObrainStatus obrain_find_nodes_by_property(ObrainDatabase* db, const char* property, const char* value_json, uint64_t** out_ids, size_t* out_count);
void         obrain_free_node_ids(uint64_t* ids, size_t count);

/* ---- Vector operations --------------------------------------------------- */

ObrainStatus obrain_create_vector_index(ObrainDatabase* db, const char* label, const char* property, int32_t dimensions, const char* metric, int32_t m, int32_t ef_construction);
int32_t      obrain_drop_vector_index(ObrainDatabase* db, const char* label, const char* property);
ObrainStatus obrain_rebuild_vector_index(ObrainDatabase* db, const char* label, const char* property);
ObrainStatus obrain_vector_search(ObrainDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
ObrainStatus obrain_mmr_search(ObrainDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t fetch_k, float lambda, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
ObrainStatus obrain_batch_create_nodes(ObrainDatabase* db, const char* label, const char* property, const float* vectors, size_t vector_count, size_t dimensions, uint64_t** out_ids);
void         obrain_free_vector_results(uint64_t* ids, float* distances, size_t count);

/* ---- Statistics ---------------------------------------------------------- */

size_t obrain_node_count(ObrainDatabase* db);
size_t obrain_edge_count(ObrainDatabase* db);

/* ---- Transactions -------------------------------------------------------- */

ObrainTransaction* obrain_begin_transaction(ObrainDatabase* db);
ObrainTransaction* obrain_begin_transaction_with_isolation(ObrainDatabase* db, int32_t isolation);
ObrainResult*      obrain_transaction_execute(ObrainTransaction* tx, const char* query);
ObrainResult*      obrain_transaction_execute_with_params(ObrainTransaction* tx, const char* query, const char* params_json);
ObrainStatus       obrain_commit(ObrainTransaction* tx);
ObrainStatus       obrain_rollback(ObrainTransaction* tx);
void               obrain_free_transaction(ObrainTransaction* tx);

/* ---- Admin --------------------------------------------------------------- */

char*        obrain_info(ObrainDatabase* db);
ObrainStatus obrain_save(ObrainDatabase* db, const char* path);
ObrainStatus obrain_wal_checkpoint(ObrainDatabase* db);

/* ---- Memory management --------------------------------------------------- */

void obrain_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* OBRAIN_H */

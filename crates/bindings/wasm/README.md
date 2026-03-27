# @obrain-db/wasm

Low-level WebAssembly binary for [Obrain](https://github.com/ObrainDB/obrain), a high-performance graph database.

## Which Package Do You Need?

| Package | Use Case |
|---------|----------|
| [`@obrain-db/web`](https://www.npmjs.com/package/@obrain-db/web) | Browser apps with IndexedDB, Web Workers, React/Vue/Svelte (recommended) |
| [`@obrain-db/wasm`](https://www.npmjs.com/package/@obrain-db/wasm) | Raw WASM binary for custom loaders or non-standard runtimes |
| [`@obrain-db/js`](https://www.npmjs.com/package/@obrain-db/js) | Node.js native bindings (faster than WASM for server-side) |

**Most users should use `@obrain-db/web`** - it wraps this package and adds browser-specific features.

## Installation

```bash
npm install @obrain-db/wasm
```

## Usage

```typescript
import init, { Database } from '@obrain-db/wasm';

// Initialize the WASM module
await init();

// Create a database and query
const db = new Database();
const result = db.execute(`MATCH (n:Person) RETURN n.name`);
```

## Status

- [x] Core WASM bindings via wasm-bindgen
- [x] In-memory database support
- [x] GQL query language (default via `browser` profile)
- [x] TypeScript type definitions
- [x] Size optimization (513 KB gzipped lite, 531 KB AI variant)
- [x] Vector search bindings (k-NN, MMR)
- [x] Snapshot export/import for IndexedDB persistence
- [x] Batch import (importLpg, importRdf, importRows)
- [x] Memory introspection (memoryUsage)

## Package Contents

```
@obrain-db/wasm/
├── obrain_wasm_bg.wasm    # WebAssembly binary
├── obrain_wasm.js         # JavaScript loader
├── obrain_wasm.d.ts       # TypeScript definitions
└── package.json
```

## Bundle Size

| Build | Size (gzip) |
|-------|-------------|
| AI variant (GQL + vector/text/hybrid search) | 531 KB |
| Lite variant (GQL only) | 513 KB |

## Runtime Support

| Runtime | Status |
|---------|--------|
| Browser (Chrome, Firefox, Safari, Edge) | Supported |
| Deno | Supported |
| Cloudflare Workers | Untested |
| Node.js | Use `@obrain-db/js` instead |

## Links

- [Documentation](https://obrain.dev)
- [GitHub](https://github.com/ObrainDB/obrain)
- [Roadmap](https://github.com/ObrainDB/obrain/blob/main/docs/roadmap.md)

## License

Apache-2.0

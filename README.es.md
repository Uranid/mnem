<div align="center">

<img src="assets/logo/mnem-banner.svg" alt="mnem: Git for Knowledge Graphs" />

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Uranid/mnem/ci.yml?style=for-the-badge&label=CI)](https://github.com/Uranid/mnem/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnem-cli?style=for-the-badge)](https://crates.io/crates/mnem-cli)
[![PyPI](https://img.shields.io/pypi/v/mnem-cli?style=for-the-badge)](https://pypi.org/project/mnem-cli/)
[![npm](https://img.shields.io/npm/v/mnem-cli?style=for-the-badge)](https://www.npmjs.com/package/mnem-cli)
[![MSRV 1.95](https://img.shields.io/badge/MSRV-1.95-orange?style=for-the-badge)](rust-toolchain.toml)
[![Runs on Linux macOS Windows WASM](https://img.shields.io/badge/runs%20on-linux%20%7C%20macos%20%7C%20windows%20%7C%20wasm-2ea44f?style=for-the-badge)](#instalación)

</div>

<div align="center">

[English](README.md) &nbsp;·&nbsp; [中文](README.zh-CN.md) &nbsp;·&nbsp; [Español](README.es.md)

</div>

<hr>

<div align="center">

https://github.com/user-attachments/assets/bd744a7e-8e89-4531-bd96-fdee0030c390

</div>

<hr>

> [!NOTE]
> Este documento es una traducción comunitaria. Para el contenido más reciente, consulta el [README en inglés](README.md).

1. [El problema](#el-problema)
2. [Benchmarks](#benchmarks)
3. [Instalación](#instalación)
4. [Inicio rápido](#inicio-rápido)
5. [Integrar](#mnem-integrate---integrar-en-cualquier-host-de-agente)
6. [Qué es](#qué-es)
7. [Comandos](#comandos)
8. [API de Python (mnem-py)](#api-de-python-mnem-py)
9. [GraphRAG](#graphrag)
10. [vs otros](#comparado-con-otros)
11. [Documentación](#documentación)
12. [Contribuir](#contribuir)

<hr>

## El problema

> **Una transcripción no es una memoria.**

Ya tienes un modelo mental para esto: git. Commits con historial que puedes comparar y revertir, ramas que puedes fusionar, un registro de cada decisión y su motivo. La memoria de tu agente no te ofrece nada de eso. Es una transcripción pegada de vuelta en el prompt, o un índice de búsqueda que no puedes inspeccionar ni editar. Las convenciones viven en archivos `.cursorrules` planos - útiles, pero no consultables ni versionables. Y las sesiones están aisladas: planifica una migración con Claude Code hoy, abre Cursor mañana, y ese agente nunca habrá oído hablar de ella.

mnem lleva el modelo de git al conocimiento de los agentes. Cada escritura es un **commit con direccionamiento por contenido** - mismos bytes, mismo CID, cualquier máquina. Las habilidades, decisiones y notas viven en un **grafo versionado, con soporte de ramas y fusionable**: `diff` para ver qué cambió entre sesiones, `revert` para deshacer un lote de hechos incorrectos, `merge` para combinar el conocimiento de dos agentes igual que fusionarías una rama.

La recuperación es **híbrida y transparente**: vector + palabras clave + recorrido de grafo en un solo paso, con un presupuesto de tokens explícito - mnem informa exactamente qué encontró, qué omitió y cuántos tokens se utilizaron. **Cambia el embedder, el reranker o el LLM con una sola línea de configuración.** Un único `mnem integrate` lo conecta a Claude Code, Cursor, Codex, Gemini CLI o cualquier host MCP. Binario único de ~40 MB. Sin daemon, sin nube, sin claves de API.

> Cierra el portátil. Ábrelo mañana. Tu agente recuerda.

<hr>

## Benchmarks

**Medido cara a cara contra mem0 y MemPalace en seis conjuntos de datos públicos. mnem lidera en todos ellos.**

Embedder ONNX MiniLM-L6-v2, mismos bytes en todos los sistemas. Sin reranking con LLM. Reproducir: `bash benchmarks/harness/run_bench.sh`.

<div align="center"><img src="assets/benchmarks/benchmarks.svg" alt="mnem public benchmarks" /></div>

<sup>Columnas de mem0: nuestra reproducción bajo el mismo arnés (mem0 no publica titulares de R@K en estos conjuntos de datos). Columnas de MemPalace: números de titulares públicos verificados cruzadamente bajo nuestro arnés. Artefactos brutos: [`benchmarks/results/v0.1.0/`](benchmarks/results/v0.1.0/). † FinanceBench usa Ollama bge-large (1024 dimensiones) en todos los sistemas; MemPalace se muestra en la mejor configuración (bge-large con ChromaDB directo); mem0 aplica extracción de memoria con LLM antes del almacenamiento. Metodología completa: [`benchmarks/results/analysis/financebench.md`](benchmarks/results/analysis/financebench.md).</sup>

### Velocidad de consulta

<div align="center"><img src="assets/benchmarks/query-speed.svg" alt="mnem query speed" /></div>

<details>
<summary><b>Reproducir</b></summary>

```bash
mnem bench fetch longmemeval     # download datasets (one-time, 264 MB)
mnem bench                       # TUI; select benchmarks interactively
mnem bench run --benches longmemeval --limit 50 --non-interactive
mnem bench results ./bench-out   # re-render results from a prior run

# Legacy bash harness (canonical path for headline numbers)
bash benchmarks/harness/run_bench.sh
```

Metodología, artefactos brutos, desglose por benchmark: [`benchmarks/`](benchmarks/) y [`docs/src/benchmarks/`](docs/src/benchmarks/).

</details>

<hr>

## Instalación

**Elige el que ya tengas. Cualquiera funciona.** Notas completas por plataforma a continuación.

```bash
# if you have Cargo (Rust): recommended for dev machines
cargo install --locked mnem-cli --features bundled-embedder

# if you have pip (Python)
pip install mnem-cli

# if you have npm (Node.js)
npm install -g mnem-cli
```

```bash
mnem --version    # confirm install
```

> [!NOTE]
> `--features bundled-embedder` incluye un ONNX MiniLM-L6-v2 en proceso para que `mnem retrieve` funcione sin ninguna configuración. Omite el indicador si prefieres usar tu propio embedder (Ollama, OpenAI, Cohere) a través de `.mnem/config.toml`.

<details>
<summary><b>macOS / Linux</b></summary>

¿No tienes Cargo? [Instala mediante rustup](https://rustup.rs/) (también instala `rustc`).

```bash
# C++ stdlib required to link the bundled ONNX Runtime (Linux only)
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
cargo install --locked mnem-cli --features bundled-embedder

# CUDA-accelerated embedder (Linux, NVIDIA GPU)
cargo install --locked mnem-cli --features bundled-embedder-cuda
```

Si `mnem` no se encuentra tras la instalación, `~/.cargo/bin` no está en `$PATH`.

**Instalación con rustup**: carga el entorno (o abre una nueva terminal):
```bash
source ~/.cargo/env
```

**Rust del sistema (apt/dnf)**: añade a PATH de forma permanente:
```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

</details>

<details>
<summary><b>Windows</b></summary>

¿No tienes Cargo? [Instala mediante rustup](https://rustup.rs/) (también instala `rustc`).

```powershell
cargo install --locked mnem-cli --features bundled-embedder

# DirectML-accelerated embedder (any GPU vendor on Windows)
cargo install --locked mnem-cli --features bundled-embedder-directml
```

</details>

<details>
<summary><b>npm / Node.js</b></summary>

¿No tienes npm? [Instala Node.js](https://nodejs.org/en/download) (npm viene incluido, se requiere Node 18+).

```bash
npm install -g mnem-cli
mnem --version

# or without a global install (one-shot)
npx mnem-cli --version
```

Descarga el binario nativo precompilado para tu plataforma en el momento de la instalación. Se requiere Node 18+. Embedder incluido - no se necesita Ollama ni clave de API.

</details>

<details>
<summary><b>pip (PyPI)</b></summary>

¿No tienes pip? [Instala Python](https://www.python.org/downloads/) (pip viene incluido con Python 3.4+).

```bash
pip install mnem-cli
mnem --version
```

Distribuye el binario `mnem` como una wheel para manylinux / macOS / Windows con el embedder incluido preintegrado.

</details>

<details>
<summary><b>Docker</b></summary>

¿No tienes Docker? [Instala Docker Desktop](https://docs.docker.com/get-started/get-docker/).

```bash
docker run --rm -p 9876:9876 ghcr.io/uranid/mnem:latest http serve
```

La imagen incluye el embedder integrado. Ejecuta `mnem mcp` dentro del contenedor para acceder a la interfaz del servidor MCP.

</details>

<details>
<summary><b>Desde el código fuente</b></summary>

```bash
# C++ stdlib required to link the bundled ONNX Runtime (Linux only)
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
git clone https://github.com/Uranid/mnem
cd mnem
cargo install --path crates/mnem-cli --features bundled-embedder
```

Requiere Rust 1.95+. Si es necesario: `rustup install 1.95 && rustup default 1.95`.

</details>

```bash
mnem --version
mnem doctor        # checks embedder + store + config, prints a green/yellow/red checklist
```

Matriz de instalación completa: [`docs/src/install.md`](docs/src/install.md).

> **¿Integrando mnem dentro de una aplicación Python?** El `pip install mnem-cli` anterior instala el **binario CLI** como una wheel. La **API nativa de Python** (`import mnem`) se encuentra en un paquete separado. Ve a **[API de Python (mnem-py) ↓](#api-de-python-mnem-py)** para `pip install mnem-py` y ejemplos de uso.

<hr>

## Inicio rápido

```bash
mkdir my-graph && cd my-graph
mnem init
mnem ingest README.md
mnem retrieve "what does this project do"
```

Cinco minutos desde cero. Consulta [`docs/src/quickstart.md`](docs/src/quickstart.md) para el recorrido completo.

<hr>

## `mnem integrate` - integrar en cualquier host de agente

Un solo comando integra la **entrada del servidor MCP**, el **hook UserPromptSubmit** (para hosts que lo admiten) y el **prompt de sistema de mnem** en el archivo de reglas del proyecto del host. Reinicia el host y el agente comienza a usar mnem automáticamente.

```bash
mnem integrate                           # interactive: detect installed hosts and prompt
mnem integrate claude-code               # wire a specific host, skip interactive detection
mnem integrate --all                     # wire every detected host without prompting

mnem integrate --check                   # report wired state for all hosts; nothing changes
mnem integrate --dry-run                 # preview what would be written without changing anything
mnem integrate --show claude-code        # print the MCP JSON block for manual copy-paste

mnem integrate --no-hooks                # skip UserPromptSubmit hook wiring
mnem integrate --no-system-prompt        # skip system prompt wiring
mnem integrate --target-repo ~/notes     # point the MCP server at a specific graph, not the global one
```

**Qué se integra:**
- **Servidor MCP** (`mcpServers.mnem`) - el agente obtiene acceso completo a las herramientas de mnem mediante `mnem mcp --repo <graph>`; por defecto apunta al grafo global (`~/.mnemglobal/.mnem`)
- **Hook UserPromptSubmit** (solo Claude Code) - ejecuta `mnem retrieve` antes de cada mensaje, inyectando automáticamente la memoria relevante en el contexto
- **Prompt de sistema** - instrucciones de uso de mnem inyectadas en el archivo de reglas del proyecto del host

El hook consulta primero el directorio `.mnem/` de tu proyecto (recorriendo hacia arriba desde el directorio actual) y luego recurre automáticamente a `mnem global retrieve`. El hook y el prompt de sistema funcionan igual independientemente del grafo de conocimiento predeterminado que elijas durante la configuración. Usa `--target-repo` solo si quieres que el servidor MCP apunte a un lugar distinto del grafo global.

Detecta y configura automáticamente:
- Claude Code
- Claude Desktop
- Cursor
- Continue
- Zed
- Gemini CLI

Cualquier otro host compatible con MCP funciona mediante una entrada `mcpServers` editada manualmente que apunte a `mnem mcp --repo <path>` - consulta [`docs/src/mcp.md`](docs/src/mcp.md).

El agente obtiene el conjunto completo de herramientas de mnem como herramientas nativas: recuperar, confirmar, ingerir, tombstone, recorrer, acceso al grafo global y más. Sin daemon adicional, sin puertos que gestionar. Referencia completa de herramientas: [`docs/src/mcp.md`](docs/src/mcp.md).

<hr>

## Qué es

**Un grafo de conocimiento con direccionamiento por contenido, recuperación híbrida GraphRAG, commits versionados e ingestión determinista, construido como sustrato de memoria persistente para agentes de IA.**

Cada nodo lleva una identidad criptográfica derivada de DAG-CBOR + BLAKE3: el mismo contenido produce el mismo CID en cualquier máquina. La recuperación combina vectorial (HNSW), dispersa (BM25/SPLADE) y recorrido multi-salto del grafo mediante RRF en un solo paso, y cada respuesta informa exactamente qué candidatos se evaluaron y cuáles se descartaron según tu presupuesto de tokens. La ingestión no requiere LLM. Binario único. Sin nube. Compila a `wasm32`.

## Qué obtienes

Cada elemento a continuación presenta primero el beneficio en lenguaje claro y luego el detalle técnico.
Etiquetas: <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> = exclusivo de mnem en memoria para agentes hoy &nbsp;·&nbsp; <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> = poco común (1-2 competidores, generalmente de pago) &nbsp;·&nbsp; (sin etiqueta) = estándar, bien implementado.

### Memoria que funciona como git

- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **Crea ramas, compara diferencias y fusiona, como git, pero para lo que sabe tu agente.** Cada escritura es un commit versionado con historial firmado Ed25519. Dos agentes (o dos máquinas) que escriben en el mismo ámbito de forma desconectada reconcilian sus cambios mediante una fusión de grafo + embeddings a 3 vías, no por "gana el último en escribir". → [Conceptos fundamentales](docs/src/core-concepts.md)
- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **La misma entrada siempre llega a la misma dirección, en cualquier computadora.** Cada nodo, árbol, sidecar y commit se direcciona por contenido mediante DAG-CBOR + BLAKE3 canónico. El contenido idéntico colapsa en un único CID. El determinismo y la reproducibilidad son propiedades garantizadas, no solo un eslogan. → [Conceptos fundamentales](docs/src/core-concepts.md)
- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **Las habilidades se convierten en un grafo consultable, no en markdown plano.** Reemplaza `AGENTS.md` y `.cursorrules` con un grafo versionado, con ramas, fusionable. Exporta tu grafo, importa el de un compañero, compara los dos, fusiona las partes que quieras.

### Recuperación que muestra su proceso

- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **Nada desaparece en silencio al agotar tu presupuesto de tokens.** Cada recuperación emite `tokens_used`, `candidates_seen` y contadores `dropped` como campos de respuesta de primera clase. Ningún otro sistema de memoria para agentes expone esto.
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Recuperación de primer nivel en todos los benchmarks públicos.** Supera a los competidores de código abierto por **+0.218 R@5 en LoCoMo**, **+0.120 en MemBench**, **+0.047 en ConvoMem** con el mismo embedder. Iguala a MemPalace en LongMemEval (R@5 0.966). Todos los números son reproducibles con el harness incluido. → [Benchmarks](#benchmarks)
- **Busca por significado, por palabra clave y por relación, en un solo paso.** GraphRAG híbrido: vectorial (HNSW) + disperso (BM25/SPLADE) + recorrido multi-salto del grafo, fusionados mediante RRF. El recorrido del grafo es opcional: activo cuando los saltos múltiples ayudan, inactivo cuando la densidad vectorial es suficiente.

### Diseñado para ejecutarse en cualquier lugar

- <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> &nbsp; **Se ejecuta en una pestaña del navegador.** `mnem-core` no tiene tokio, ni sistema de archivos, ni red. El mismo código de recuperación compila sin cambios a `wasm32`: Chrome, Cloudflare Workers, arranque en frío de Lambda. Graphiti y mem0 son stacks de Python + base de datos externa; no pueden desplegarse en el edge.
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Un único binario de ~40 MB. Sin daemon, sin nube, sin cuenta.** Almacén redb integrado, funciona completamente sin conexión. La misma imagen impulsa tanto la CLI como el servidor HTTP.
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Listo para usar en segundos.** El modelo ONNX MiniLM-L6-v2 incluido se ejecuta en proceso: sin Ollama, sin claves de API, sin llamada de red en el arranque. Solo `mnem init` y ya puedes recuperar. mem0 y Graphiti necesitan un endpoint LLM externo durante la ingestión. → [Instalación](docs/src/install.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Cambia cualquier proveedor con una línea de configuración.** El embedder, el codificador disperso, el reranker y el LLM son todos configurables. Pasa de ONNX local a Cohere alojado con un solo flag. Sin forks, sin recompilación. → [Proveedores de embeddings](docs/src/guides/embed-providers.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Un motor, cuatro puertas de entrada.** CLI, HTTP, MCP y Python envuelven el mismo motor. `mnem integrate` conecta el servidor MCP en Claude Code, Cursor, Codex, Gemini CLI, cualquier herramienta que hable MCP. → [Referencia de la CLI](docs/src/cli.md) &nbsp;·&nbsp; [MCP](docs/src/mcp.md)

### Señales de confianza

- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Los mismos bytes de entrada siempre producen los mismos CIDs de salida.** Ingestión determinista: sin LLM durante la ingestión; el análisis, fragmentación y extracción son estadísticos (KeyBERT opcional). Apto para auditorías, probado con fuzzing, byte a byte idéntico entre máquinas. → [Pipeline de ingestión](docs/src/guides/ingest.md)
- <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> &nbsp; **Probado con propiedades y fuzzing a nivel de infraestructura de base de datos.** Los parsers se prueban por propiedades y se someten a fuzzing; el viaje de ida y vuelta CAR y los merge-commits son byte a byte idénticos. Una señal de confianza que normalmente solo se ve en bases de datos fundacionales.

### Cuándo mnem es la opción adecuada

- El conocimiento se acumula a lo largo de muchas sesiones y alguien necesita razonar sobre el historial.
- Dos agentes (o dos máquinas) editan la misma memoria y necesitan reconciliar los cambios de forma limpia.
- Las auditorías importan: misma entrada, misma salida, reproducible en cualquier computadora.
- El despliegue es en el edge, sin conexión o en un entorno aislado (navegador, Cloudflare Workers, arranque en frío de Lambda).

<hr>

## Comandos

Cada comando acepta `--help` para la referencia completa de flags.

### Inicialización y salud

```bash
mnem init      # create a new graph in the current directory
mnem doctor    # probe embedder + store + config; green/yellow/red checklist
mnem stats     # nodes, edges, refs, embedder health, repo size
```

### Agregar conocimiento

```bash
mnem ingest notes.md                        # parse a file into Doc + Chunk + Entity nodes
mnem ingest --recursive docs/               # ingest a directory recursively
mnem ingest --chunker recursive report.pdf  # PDF with sliding-window chunking
```

```bash
mnem add node -s "Alice leads the infra team"                       # label defaults to "Node"
mnem add node --label Fact -s "Alice leads the infra team"          # add a single fact node
mnem add edge --from <uuid> --to <uuid> --label works_at            # connect two nodes
```

```bash
mnem get <uuid>                                                     # fetch a node by UUID: ntype, summary, props
mnem get <uuid> --content                                           # also print the full content body
mnem tombstone <uuid>                                               # soft-delete: excluded from retrieval, kept in audit log
mnem tombstone <uuid> --reason "superseded by newer decision"       # with reason recorded in op-log
mnem delete <uuid>                                                  # hard-delete: no audit trail
mnem global get <uuid>                                              # look up a node in the global graph
mnem global tombstone <uuid>                                        # tombstone a node in the global graph
```

> El pipeline de ingestión es determinista: sin LLM en el momento de la ingestión, los mismos bytes de entrada siempre producen los mismos CIDs de salida. Amigable para auditorías y con pruebas fuzz.

### Recuperar conocimiento

```bash
mnem retrieve "what did we decide about the API design"  # searches local .mnem/ first, falls back to global
mnem -R ~/notes retrieve "query"                         # target a specific graph explicitly
```

`-R <path>` es un flag global que redirige cualquier comando a un directorio de repositorio específico. Anula la búsqueda ascendente desde el directorio actual y cualquier valor predeterminado establecido mediante `mnem integrate`. Se aplica a todos los subcomandos: `mnem -R ~/notes status`, `mnem -R ~/notes log`, etc.

Recuperación híbrida: vectorial (HNSW) + dispersa (BM25/SPLADE) + recorrido del grafo, fusionados mediante RRF. Consulta [GraphRAG](#graphrag) para los flags de ajuste.

### El grafo global

> [!NOTE]
> mnem tiene dos ámbitos: el **grafo local** (`.mnem/` en el directorio de tu proyecto) y el **grafo global** (`~/.mnemglobal/.mnem/`). El grafo global es para hechos entre proyectos y entre sesiones que deben seguirte a todas partes.

**Cuándo usar local vs global:**

| Usa `.mnem/` local para | Usa `mnem global` para |
|------------------------|----------------------|
| Hechos, decisiones y contexto de código específicos del proyecto | Personas, preferencias y hechos que abarcan todos los proyectos |
| Memoria por repositorio que viaja con el repositorio | Conocimiento que quieres que vean todas las sesiones y todos los agentes |
| Todo lo que incluirías en un commit junto al código | Continuidad entre sesiones |

`mnem global` es un espejo completo de `mnem` pero opera exclusivamente sobre el grafo global:

```bash
mnem global retrieve "what is Alice's current role"     # search the global graph only
mnem global ingest contacts.md                          # ingest a file into the global graph
mnem global add node --label Entity:Person \
  --prop name=Alice -s "Alice leads the infra team"     # add a node to the global graph
```

El comando `mnem integrate` configura el agente para leer primero el grafo local y recurrir automáticamente al global - no se requiere cambio manual durante el uso normal.

### Estado e inspección

```bash
mnem status           # op-head CID, head commit, all named refs, label counts, MERGING marker
mnem stats            # one-line: op, commit, content CID, ref count, label names
```

### Historial

```bash
mnem log              # walk op-log backwards from HEAD, default last 20 entries
mnem log -n 50        # show last 50 entries
mnem log --oneline    # compact one-line-per-op format
mnem log --format json # machine-readable JSON stream

mnem show             # decode and pretty-print the current op-head block
mnem show <cid>       # decode any block by CID (Node, Edge, Commit, Operation, View, ...)

mnem diff <op-a-cid> <op-b-cid>   # ref deltas + node/edge structural diff between two ops
mnem diff HEAD <cid>               # diff current op against a specific op CID
```

### Ramas y fusiones

```bash
mnem branch list                        # list all refs/heads/* branches; * marks current
mnem branch create <name>               # create branch at current HEAD
mnem branch create <name> <start>       # branch from a ref name, branch name, or CID
mnem branch create <name> --from HEAD   # explicit --from form; same resolution as above
mnem branch delete <name>               # delete a local branch pointer

mnem merge <branch>                     # 3-way merge <branch> into current HEAD
mnem merge <branch> --strategy=ours     # auto-resolve conflicts: keep current side
mnem merge <branch> --strategy=theirs   # auto-resolve conflicts: take incoming side
mnem merge <branch> --dry-run           # preview outcome without persisting anything
mnem merge --continue                   # finish after editing .mnem/MERGE_CONFLICTS.json
mnem merge --abort                      # cancel, restore HEAD from .mnem/ORIG_HEAD

mnem pull                               # fast-forward origin/main into HEAD (default)
mnem pull <remote> <branch>             # fast-forward <remote>/<branch> into HEAD
```

### Operaciones remotas

```bash
mnem remote add <name> <url>            # register a remote (stores in .mnem/config.toml)
mnem remote add <name> <url> \
  --token-env MNEM_REMOTE_ORIGIN_TOKEN  # name the env var that holds the bearer token
mnem remote list                        # list all configured remotes with their URLs
mnem remote show <name>                 # show URL + capabilities for one remote
mnem remote remove <name>               # remove a remote entry

mnem fetch                              # fetch from origin (default)
mnem fetch <remote>                     # fetch from a named remote; token via env var

mnem push                               # push HEAD to origin/main (default)
mnem push <remote> <branch>             # push a specific branch to a named remote

mnem clone <url> [<dir>]                # clone a CAR archive into <dir>; file:// and bare .car paths supported
mnem clone file:///tmp/repo.car ./copy  # clone from a local file URL
mnem clone ./repo.car ./copy            # bare path shorthand (must end in .car)
```

### Consulta y recorrido del grafo

```bash
mnem query --where name=Alice                    # exact property match, default 10 results
mnem query --where kind=Person -n 25             # increase result limit
mnem query --where kind=Person \
  --with-outgoing knows                          # match nodes + follow outgoing "knows" edges
mnem query --where status=active \
  --with-outgoing depends_on \
  --with-outgoing depends_on                     # repeat --with-outgoing to chain hops

mnem blame <node-uuid>                           # list all incoming edges to a node
mnem blame <node-uuid> --etype authored          # filter to one edge type
```

### Referencias con nombre

```bash
mnem ref list                         # list all refs (refs/heads/*, refs/remotes/*, ...)
mnem ref set <name> <target-cid>      # point a ref at a specific commit CID
mnem ref delete <name>                # delete a named ref
```

### Incrustaciones (embeddings)

```bash
mnem embed                            # backfill embeddings for every node missing a vector
mnem embed --force                    # re-embed even nodes that already have a vector
mnem embed --label Person             # restrict to nodes of one label
mnem embed --dry-run                  # count what would be embedded without calling the provider

mnem reindex                          # alias for embed; preferred name after C7 rename
mnem reindex --label Doc              # restrict to one label
mnem reindex --since <commit>         # only nodes added/changed after <commit>
mnem reindex --force                  # re-embed already-indexed nodes
mnem reindex --dry-run                # count without calling the provider
```

### Acceso a bloques de bajo nivel

```bash
mnem cat-file <cid>          # emit raw DAG-CBOR bytes for a block to stdout
mnem cat-file <cid> --json   # decode to DAG-JSON and pretty-print (pipe into jq)
```

### Exportar e importar

```bash
mnem export <path>                        # export HEAD as a CAR v1 archive
mnem export -                             # write CAR to stdout (pipe over SSH etc.)
mnem export --from refs/heads/main out.car  # export from a specific ref
mnem export --from <cid> backup.car       # export from a specific commit CID

mnem import <path>                        # import a CAR archive into the current repo
mnem import -                             # read CAR from stdin
```

### Configuración

```bash
mnem config set user.name Alice           # set author name
mnem config set user.email alice@example.com
mnem config set embed.provider ollama     # embedder: openai | ollama
mnem config set embed.model nomic-embed-text
mnem config set embed.base_url http://localhost:11434  # override provider endpoint
mnem config get embed.provider            # print the current value of a key
mnem config unset embed.provider          # remove a key
mnem config list                          # print all set keys and their values
```

Claves conocidas: `user.name`, `user.email`, `user.key`, `user.agent_id`, `embed.provider`, `embed.model`, `embed.api_key_env`, `embed.base_url`. Las claves de API viven en variables de entorno, nunca en la configuración.

### Registro de repositorios

```bash
mnem repos list              # list all repos registered with mnem integrate
mnem repos set-default <path>  # mark a repo as the default for mnem without -R
mnem repos prune             # remove registry entries for paths that no longer exist
```

### Servidores

```bash
mnem mcp                       # start the MCP JSON-RPC server over stdio
mnem mcp --repo ~/notes        # point the MCP server at a specific graph
mnem http serve                # start the HTTP JSON API (loopback by default)
```

### Benchmarks

```bash
mnem bench                                       # interactive TUI; select benchmarks to run
mnem bench run --benches longmemeval --limit 50  # run a specific benchmark suite
mnem bench fetch longmemeval                     # download benchmark datasets
mnem bench results ./bench-out                   # re-render results from a prior run
```

### Autocompletado de shell

```bash
mnem completions bash        # emit bash completion script
mnem completions zsh         # zsh
mnem completions fish        # fish
mnem completions powershell  # PowerShell
mnem completions elvish      # Elvish

# Install (bash):
mnem completions bash > ~/.local/share/bash-completion/completions/mnem
# Install (zsh):
mnem completions zsh > ~/.zsh/completions/_mnem
```

Referencia completa de la CLI: [`docs/src/cli.md`](docs/src/cli.md).

<hr>

## API de Python (mnem-py)

Usa `mnem-py` cuando quieras leer y escribir un grafo mnem directamente desde Python - sin el binario de la CLI. El mismo motor de recuperación, con enlaces PyO3.

```bash
pip install mnem-py
pip install sentence-transformers   # brings ~200 MB of deps (torch, transformers)
```

`mnem-py` almacena y recupera mediante **vector denso**: tú calculas los embeddings en Python y se los pasas a mnem. `SentenceTransformer("all-MiniLM-L6-v2")` descarga un modelo de ~23 MB desde HuggingFace Hub la primera vez que se usa y lo almacena en caché en `~/.cache/huggingface/`; todas las llamadas posteriores son completamente locales sin necesidad de red.

```python
import pymnem
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # downloaded once, ~23 MB
MODEL_NAME = "all-MiniLM-L6-v2"                    # key mnem uses to match stored vectors

repo = pymnem.Repo.init_memory()                    # in-memory; open_or_init() for disk

# Write: compute an embedding for each node and attach it
with repo.transaction(author="agent", message="seed") as tx:
    for text in ["Alice lives in Berlin", "Bob moved to Paris"]:
        tx.add_node(ntype="Memory", summary=text)
        tx.add_embedding_f32(MODEL_NAME, model.encode(text).tolist())

# Retrieve: compute a query vector with the same model, mnem ranks under token budget
query_vec = model.encode("Alice Berlin").tolist()
result = repo.retrieve(vector=query_vec, model=MODEL_NAME, token_budget=500, limit=5)
for item in result:
    print(f"{item.score:.3f}  {item.summary}")
# result.tokens_used / result.tokens_budget  - no silent truncation
```

Superficie completa de la API - `query`, `update_node`, `delete_node`, persistencia en disco, filtrado por etiqueta: [`crates/mnem-py/README.md`](crates/mnem-py/README.md).

<hr>

## GraphRAG

mnem incluye GraphRAG integrado. Un parámetro por etapa, activación opcional por consulta, nunca obligatorio. La búsqueda vectorial por sí sola gestiona bien la mayoría de las consultas - activa las etapas de grafo cuando las consultas abarcan múltiples documentos, requieren razonamiento de múltiples saltos o necesitan respuestas composicionales.

### Etapas y opciones

| Etapa | Opción | Qué hace |
|-------|------|------|
| **Canal vectorial** | siempre activo | HNSW sobre embeddings densos por commit (MiniLM de 384 dimensiones por defecto). |
| **Canal disperso** | controlado por configuración | BM25 + SPLADE-onnx, fusionado con el vector mediante Reciprocal Rank Fusion. Activado por el bloque `[sparse]` en `config.toml`. |
| **Conjunto de candidatos vectoriales** | `--vector-cap <N>` | Amplía el tamaño del conjunto denso desde el valor por defecto de 256. Mayor valor = mejor recuperación de cola larga, con mayor coste. |
| **Límite de resultados** | `--limit <N>` | Conjunto final devuelto (por defecto 10). Forma abreviada: `-n`. |
| **Expansión de grafo** | `--graph-expand <N>` | Añade N vecinos de las semillas top-K mediante aristas de autoría. Valor por defecto recomendado para auditoría: `20` cuando el grafo está activo. |
| **Modo de grafo** | `--graph-mode <decay\|ppr>` | `decay` (por defecto) = peso exponencial por salto. `ppr` = Personalised PageRank sobre el índice de adyacencia híbrido, puntuación de calidad académica para múltiples saltos. |
| **Filtro de comunidad** | `--community-filter` | Ejecuta la detección de comunidades Leiden; descarta comunidades de baja cobertura antes de la fusión. Umbral de cobertura por defecto: `0.5`. |
| **Extracción KeyBERT** | `mnem ingest --extractor keybert` | Enriquecimiento de frases clave en el momento de la ingesta. Refuerza las señales dispersas y de comunidad. Se aplica en la ingesta, no en la recuperación. |
| **Resumen** | `--summarize` | Resumen de centroide + MMR del top-K, con diversidad. |
| **Reordenación con codificador cruzado** | `--rerank <provider:model>` | Reordenación posterior a la fusión. Compatible con `cohere:rerank-english-v3.0`, `voyage:rerank-1`, local. |

### Ejemplos rápidos

```bash
# Dense baseline
mnem retrieve "what does this project do"

# Add multi-hop graph traversal
mnem retrieve "..." --graph-expand 20

# Full stack: graph-expand + community-filter + PPR + rerank
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr --rerank cohere:rerank-english-v3.0

# Stack a cross-encoder reranker on top
mnem retrieve "..." --graph-expand 20 --community-filter --rerank cohere:rerank-english-v3.0

# Ingest with KeyBERT keyphrase enrichment (strengthens sparse + community signals)
mnem ingest --extractor keybert notes.md
```

### Cuándo activar

- **Corpus de un solo documento, consultas simples**: deja el grafo desactivado, la búsqueda vectorial sola es suficiente
- **Preguntas de múltiples saltos o composicionales**: `--graph-expand 20`
- **Historial extenso con referencias entre documentos**: añade `--community-filter`
- **Techo de recuperación necesario**: apila `--rerank` encima
- **Ingesta enriquecida con frases clave**: `mnem ingest --extractor keybert` en el momento de la ingesta

Arquitectura completa de recuperación: [`docs/src/cli.md`](docs/src/cli.md) (parámetros de recuperación)

<hr>

## Comparado con otros

- [mnem vs mem0](docs/src/comparisons/mem0.md) - capa de memoria para agentes, líder en OSS
- [mnem vs MemPalace](docs/src/comparisons/mempalace.md) - par metodológico
- [mnem vs Supermemory](docs/src/comparisons/supermemory.md) - solución dominante de nube cerrada
- [mnem vs Cognee](docs/src/comparisons/cognee.md) - alternativa de KG para agentes
- [mnem vs Letta](docs/src/comparisons/letta.md) - framework de memoria para agentes
- [mnem vs graphify](docs/src/comparisons/graphify.md) - herramienta de grafo ligera

Matriz completa: [`docs/src/comparisons/README.md`](docs/src/comparisons/README.md).

<hr>

## Cuándo NO usar mnem

- **Necesitas OLTP transaccional.** mnem es de solo adición con historial versionado; la semántica de UPDATE/DELETE a nivel de fila no es el modelo.
- **Necesitas recuperación a escala de nube con menos de 50 ms a más de 10k QPS.** mnem es local primero. La recuperación fragmentada multirregión está en la hoja de ruta, no en v1.

> ¿Buscas memoria alojada, réplicas multirregión, grafos compartidos entre equipos o una capa remota gestionada? Un proyecto hermano que aporta todo eso a mnem está en desarrollo activo - mantente atento.

<hr>

## Crates

| Crate | Función |
|-------|------|
| [`mnem-cli`](crates/mnem-cli) | Binario `mnem` - un comando para todo |
| [`mnem-core`](crates/mnem-core) | Modelo de grafo, recuperación, indexación, sidecars |
| [`mnem-http`](crates/mnem-http) | Servidor HTTP JSON |
| [`mnem-mcp`](crates/mnem-mcp) | Servidor MCP (stdio) |
| [`mnem-py`](crates/mnem-py) | Enlaces Python PyO3 |
| [`mnem-embed-providers`](crates/mnem-embed-providers) | ONNX integrado, Ollama, OpenAI, Cohere |
| [`mnem-sparse-providers`](crates/mnem-sparse-providers) | BM25, SPLADE-onnx |
| [`mnem-rerank-providers`](crates/mnem-rerank-providers) | Cohere, Voyage |
| [`mnem-llm-providers`](crates/mnem-llm-providers) | OpenAI, Anthropic, Ollama |
| [`mnem-ingest`](crates/mnem-ingest) | Pipeline de análisis, fragmentación y extracción |
| [`mnem-extract`](crates/mnem-extract) | Extracción de entidades (KeyBERT, NER estadístico) |
| [`mnem-ner-providers`](crates/mnem-ner-providers) | Rasgo de proveedor NER + proveedores integrados (`RuleNer`, `NullNer`) |
| [`mnem-bench`](crates/mnem-bench) | Arnés de benchmarks (LongMemEval, LoCoMo, etc.) |
| [`mnem-graphrag`](crates/mnem-graphrag) | Resumen de comunidades, centroide + MMR |
| [`mnem-ann`](crates/mnem-ann) | Envoltorio HNSW |
| [`mnem-backend-redb`](crates/mnem-backend-redb) | Almacén respaldado por redb |
| [`mnem-transport`](crates/mnem-transport) | Codec CAR + encuadre remoto |

<hr>

## Documentación

- [Inicio rápido](docs/src/quickstart.md) - recorrido de cinco minutos
- [Instalación](docs/src/install.md) - matriz de instalación por plataforma
- [Referencia de la CLI](docs/src/cli.md) - cada subcomando y parámetro
- [Servidor MCP](docs/src/mcp.md) - herramientas expuestas, configuración del cliente
- [Conceptos fundamentales](docs/src/core-concepts.md) - CIDs, commits, etiquetas
- [Configuración](docs/src/configuration.md) - variables de entorno, config.toml
- [Metodología de benchmarks](docs/src/benchmarks/methodology.md)
- [Reproducir benchmarks](docs/src/benchmarks/reproduce.md)
- [Proveedores de embeddings](docs/src/guides/embed-providers.md)
- [Migraciones](docs/src/migrations/)

<hr>

## Contribuir

Las issues y los PRs son bienvenidos. Empieza aquí:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - convenciones de ramas, etiqueta de revisión, cómo enviar un PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - normas de participación (Contributor Covenant 2.1)
- [`SECURITY.md`](SECURITY.md) - política de divulgación de vulnerabilidades

## Licencia

[Apache-2.0](LICENSE). Consulta [`NOTICE`](NOTICE) para las atribuciones de terceros.

<hr>

## Desconectar / eliminar

```bash
mnem unintegrate                  # interactive: pick which hosts to remove mnem from
mnem unintegrate claude-code      # remove one host
mnem unintegrate --all            # remove all wired hosts
```

Ejecuta `mnem unintegrate --help` para ver todas las opciones.

<hr>

⭐ **¿Te resulta útil mnem?** Una estrella es la señal más potente que recibimos de un desarrollador satisfecho - ayuda al próximo desarrollador de agentes a encontrar este repositorio cuando tenga problemas con la memoria. Leemos cada issue, cada PR, cada mención. Cuéntanos qué construiste.

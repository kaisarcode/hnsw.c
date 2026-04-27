# Agent Protocol: hnsw.c

## 1. Structure
src/hnsw.c — CLI/main only
src/libhnsw.c — implementation only
src/hnsw.h — public API only
test.sh — external validation
bin/{arch}/{platform}/ — artifacts, committed, Git LFS
.build/ — temp, gitignored

artifacts-per-target: executable + libhnsw.a + libhnsw.so (or .dll/.dll.a on windows)
same-layout-for-libs-and-apps: no exceptions

## 2. Rules
api:no-invent — README header CLI tests impl must match. placeholders marked explicitly.
build:no-march-native-default — native opt behind HNSW_NATIVE=OFF
threading:build-then-query — single writer phase, concurrent readers after kc_hnsw_build()
lifecycle:document-ownership — who owns pointer, how long valid, safe to close while active?
scope:no-redesign — no frameworks hidden services background workers global state unless asked. return modified files only.

## 3. Validation
before-returning: README matches CLI+API+build+threading. tests cover failure cases. no inline comments inside functions.

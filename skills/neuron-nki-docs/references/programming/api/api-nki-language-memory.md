# NKI Language - Memory Operations

> **Module**: nki.language
> **Total Functions**: 5

## Overview

Memory management and data movement functions for loading/storing tensors.

## Functions

### nki.language.hbm {#nki-language-hbm}

`nki.language.hbm()`

**Signature:**

```python
language.hbm
```

HBM - Alias of private_hbm

---

### nki.language.private_hbm {#nki-language-private_hbm}

`nki.language.private_hbm()`

**Signature:**

```python
language.private_hbm
```

HBM - Only visible to each individual kernel instance in the SPMD grid

---

### nki.language.psum {#nki-language-psum}

`nki.language.psum()`

**Signature:**

```python
language.psum
```

PSUM - Only visible to each individual kernel instance in the SPMD grid

---

### nki.language.sbuf {#nki-language-sbuf}

`nki.language.sbuf()`

**Signature:**

```python
language.sbuf
```

State Buffer - Only visible to each individual kernel instance in the SPMD grid

---

### nki.language.shared_hbm {#nki-language-shared_hbm}

`nki.language.shared_hbm()`

**Signature:**

```python
language.shared_hbm
```

Shared HBM - Visible to all kernel instances in the SPMD grid

---

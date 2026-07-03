# Multiplatform Photoshop-like Application Architecture

This document outlines the architecture, code sharing, and platform-specific considerations for building a multiplatform image editing application using 
**OpenCV**, **Kotlin Multiplatform (KMP)**, and **Compose Multiplatform (CMP)**.

---

## 1. Platform Coverage

- **Desktop:** Windows, macOS, Linux
- **Mobile:** Android, iOS
- **Web:** WebAssembly (WASM)

---

## 2. Code Sharing Breakdown (100%)

| **Layer**                             | **Description**                                               | **% of Total Code** | **Notes**                                                                 |
|---------------------------------------|---------------------------------------------------------------|-------------------|--------------------------------------------------------------------------|
| **Core Image Processing (OpenCV C++)** | Filters, transformations, inpainting, edge detection, blending | 40%               | Fully shared across all platforms                                         |
| **Business Logic / Tool Management**  | Undo/redo, layers, brush sizes, filter pipelines            | 20%               | Fully shared via KMP                                                      |
| **UI Logic (CMP Compose)**            | Layouts, buttons, sliders, toolbar logic                     | 20%               | Mostly shared (~80–90%), minor tweaks per platform                        |
| **File I/O & OS Integration (KMP libs)** | Using KMP libraries like `okio`, `multiplatform-settings`    | 10%               | Shared across all platforms                                               |
| **WASM-specific Glue**                 | Canvas, memory copy, JS interop                               | 5%                | Only for Web, small glue layer                                           |
| **Camera / Sensors / Advanced OS hooks** | Native APIs for camera, notifications                        | 5%                | Platform-specific                                                         |

**Key takeaway:** With KMP libraries for file I/O and OS integration, **~95% of code is shared**, and only **~5% is platform-specific**.

---

## 3. Multiplatform Code Sharing Table (Simplified)

| Layer                               | Shareable Across Platforms | Notes                                      |
|------------------------------------|----------------------------|--------------------------------------------|
| Core Image Processing               | ✅ 100%                     | OpenCV C++ logic                           |
| Business Logic / Tool Management    | ✅ 100%                     | KMP shared logic                            |
| UI Logic (CMP Compose)              | ✅ ~80–90%                  | Mostly shared, minor tweaks                |
| File I/O & OS Integration           | ✅ 80–100%                  | Using KMP libraries like `okio`           |
| WASM-specific Glue                  | ❌ 0%                       | Canvas, JS interop, memory copy           |
| Camera / Advanced OS features       | ❌ 0–20%                     | Platform-specific                          |

---

## 4. Mermaid Diagram – Shared vs Platform-Specific Layers

```mermaid
graph TD
    A[Application] --> B[Shared Code ~95%]
    A --> C[Platform-Specific Code ~5%]

    B --> B1[Core Image Processing - OpenCV - 40%]
    B --> B2[Business Logic / Tool Management - 20%]
    B --> B3[UI Logic - CMP Compose - 20%]
    B --> B4[File I/O & OS Integration - KMP Libraries - 10%]

    C --> C1[WASM-specific Glue - Canvas, JS interop, memory - 5%]

    style B fill:#8ef1a0,stroke:#333,stroke-width:1px
    style C fill:#f19e9e,stroke:#333,stroke-width:1px
    style B1 fill:#7de0a0
    style B2 fill:#65d18e
    style B3 fill:#4cc37c
    style B4 fill:#33b469
    style C1 fill:#f08080

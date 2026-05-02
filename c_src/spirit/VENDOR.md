# Vendored Spirit sources

This directory contains a minimal, self-contained vendor of the
Vulkan compute backend from [Spirit](https://github.com/spirit-code/spirit),
included so that `nx_vulkan` can ship to hex.pm without requiring
users to clone Spirit themselves.

## What is vendored

| Path | Source in Spirit | Purpose |
|------|------------------|---------|
| `include/engine/Backend_par_vulkan.hpp` | `core/include/engine/Backend_par_vulkan.hpp` | Vulkan backend public header |
| `src/engine/Backend_par_vulkan.cpp` | `core/src/engine/Backend_par_vulkan.cpp` | Vulkan backend implementation |
| `LICENSE.txt` | `LICENSE.txt` | Spirit's MIT license — required for attribution |
| `../../priv/shaders/*.spv` | `shaders/*.spv` | Pre-compiled SPIR-V shaders (22 files) |

The vendored footprint is roughly 800 lines of C++ and 73 KB of SPIR-V.
**No other Spirit sources are required.** `Backend_par_vulkan.{hpp,cpp}`
depend only on `<vulkan/vulkan.h>` and the C++ standard library.

## Pinned upstream version

- **Repository**: https://github.com/spirit-code/spirit
- **Branch**: `feature/vulkan-backend`
- **Commit**: `136144d71865dbe2205b7ba0cd3a5ee565cac97d`
- **Date pinned**: 2026-05-02

## License

Spirit is MIT-licensed (see `LICENSE.txt` in this directory). `nx_vulkan`
itself is Apache-2.0. Vendoring MIT into Apache-2.0 is permitted; the
MIT license notice for these files must be preserved, hence
`c_src/spirit/LICENSE.txt`.

## How to refresh

If Spirit's Vulkan backend or shaders are updated upstream and you want to
roll the vendored copy forward:

```sh
# from a Spirit checkout:
cd ~/projects/learn_erl/spirit
git pull origin feature/vulkan-backend
SPIRIT_REV=$(git rev-parse HEAD)

# from the nx_vulkan checkout:
cd ~/projects/learn_erl/nx_vulkan
cp ~/projects/learn_erl/spirit/core/include/engine/Backend_par_vulkan.hpp \
   c_src/spirit/include/engine/Backend_par_vulkan.hpp
cp ~/projects/learn_erl/spirit/core/src/engine/Backend_par_vulkan.cpp \
   c_src/spirit/src/engine/Backend_par_vulkan.cpp
cp ~/projects/learn_erl/spirit/shaders/*.spv priv/shaders/
cp ~/projects/learn_erl/spirit/LICENSE.txt c_src/spirit/LICENSE.txt

# update the pinned commit in this file:
sed -i "s/^- \*\*Commit\*\*:.*/- **Commit**: \`$SPIRIT_REV\`/" c_src/spirit/VENDOR.md
sed -i "s/^- \*\*Date pinned\*\*:.*/- **Date pinned**: $(date +%Y-%m-%d)/" c_src/spirit/VENDOR.md

mix test  # verify
git add c_src/spirit priv/shaders
git commit -m "vendor: refresh Spirit to $SPIRIT_REV"
```

## Development override

Setting `SPIRIT_DIR=/path/to/spirit` at build time still refreshes
shaders from a local Spirit checkout into `priv/shaders/` on each
build — useful when iterating on shader code from mac-248. The
vendored `Backend_par_vulkan.{hpp,cpp}` are NOT swapped out by
`SPIRIT_DIR`; if you change the C++ backend code in Spirit, refresh
this directory explicitly via the procedure above.

# XDNA 4D DMA

## Why does MemTile need 4D addressing?

The XDNA MemTile DMA path does not preserve a plain row-major matrix after a
tile is packed. Step 2 writes each incoming `mct × kmt` slab as consecutive
`mct × kct` blocks. Step 3 must therefore identify both the packed `kct` tile
and the strip inside that tile.

For a strip width `s`, the required traversal has four independent indices:

```text
(b, u, i, v)
```

where `b` selects the packed `kct` tile, `u` selects the `s`-wide strip within
that tile, `i` selects a row, and `v` selects an element within the strip.

The physical address is:

```text
addr = b · (mct · kct) + u · s + i · kct + v
```

Here `b` cannot be folded into the outer strip counter: crossing a `kct` tile
changes the physical base by the full packed tile size.

## Interactive demonstration

The self-contained animation below walks through the three stages:

1. ShimTile MM2S reads a row-major `mct × K` tile.
2. MemTile S2MM packs it into consecutive `mct × kct` blocks.
3. MemTile MM2S compares the required 4D traversal with an incorrect 3D model.

<div style="width:100%; height:780px;">
  <iframe
    src="/assets/demos/xdna-4d-dma.html"
    title="Interactive XDNA 4D DMA demonstration"
    style="width:100%; height:100%; border:0; border-radius:12px;"
    loading="lazy"
    allowfullscreen>
  </iframe>
</div>

## Why 3D fails

A 3D descriptor can describe the first strip and may appear correct while the
stream remains inside one packed tile. It fails at the first `kct`-tile
boundary because it has no independent counter for the packed tile base. An
exhaustive search over possible 3D affine shapes cannot reproduce the required
address sequence for the non-degenerate proof configuration.

The demo’s **3D vs 4D** control shows the first mismatch, while **Exhaustive 3D
search** checks every possible factorization for the current parameters.

## Source

- [XDNA DMA proof animation source](/assets/demos/xdna-4d-dma.html)

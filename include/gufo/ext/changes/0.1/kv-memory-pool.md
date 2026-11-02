# Shared growable K/V memory pool

Publish the HIP K/V memory pool as experimental
(`include/gufo/ext/experimental/kv_memory.hpp`): a device address arena with
a remap quarantine that sessions map K/V chunks into as their token counts
grow, so runners can share one K/V byte budget across sessions. Additive.

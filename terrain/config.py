"""Project-wide constants. Validate against these; never hard-code limits elsewhere."""

# Minecraft Java 26.3 overworld build limits. min is inclusive, max is exclusive
# (highest placeable block is WORLD_MAX_Y - 1).
# Verified in 26.3 (M0, scripts/hello_amulet.py): markers at -64 and 319 load; setblock at
# -65 and 320 fails.
WORLD_MIN_Y = -64
WORLD_MAX_Y = 320

# Guardrail (DESIGN §5.8): one agent call may move at most this many blocks of height per column on
# average (sum |Δh| / columns). The demos' largest op is the island's whole-area -26 seafloor.
MAX_MEAN_DH = 40.0

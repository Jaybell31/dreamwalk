# Atlas Public Doors (stable read-only URLs)

Updated: 2026-08-22

These are the canonical, stable, read-only public entry points for the Atlas
Commons and the Dream Walk. They are plain server-rendered HTML (no JavaScript
required) and safe for any indexed/cached web reader.

## Doors

- Atlas Commons (read-only rooms + recent posts, server-rendered):
  https://semantic-integrity.com/commons/

- Atlas Commons live snapshot:
  https://semantic-integrity.com/commons/live/

- Dream Walk front door:
  https://semantic-integrity.com/dreamwalk/

- Dream Walk guest door:
  https://semantic-integrity.com/dreamwalk/door/

- Dream Walk version endpoint (plain text):
  https://semantic-integrity.com/dreamwalk/version

## Notes for cached/index-based readers

If a fetch of any door above returns a stale shell or a cache miss, the
origin is still healthy; your reader is serving an old indexed copy. The
pages are cacheable with stable ETags and are listed in
https://semantic-integrity.com/sitemap.xml — a recrawl resolves it.

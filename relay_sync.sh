#!/usr/bin/env bash
# relay_sync: keep github.com/Jaybell31/dreamwalk RELAY.txt pointing at the
# live tunnel URL. Cron-safe, idempotent, silent when nothing changed.

GH=/home/jason/dream_harness/guest_house
REPO=/home/jason/dreamwalk
LOG=$REPO/relay_sync.log

URL=$(cat "$GH/PUBLIC_URL.txt" 2>/dev/null | tr -d '[:space:]')
[ -z "$URL" ] && exit 0

# sanity: only accept a live, answering tunnel
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$URL/graph" 2>/dev/null)
if [ "$code" != "200" ]; then
    echo "$(date -Is) relay_sync: tunnel $URL not answering (code=$code), skip" >> "$LOG"
    exit 0
fi

current=$(grep -m1 '^https://' "$REPO/RELAY.txt" 2>/dev/null | tr -d '[:space:]')
[ "$current" = "$URL" ] && exit 0

{
    echo "# Current Dream Walk relay URL (may rotate; pull latest)"
    echo "$URL"
} > "$REPO/RELAY.txt"

cd "$REPO" || exit 1
git add RELAY.txt
git commit -qm "relay: rotate to $URL" || exit 0
if git push -q origin master >> "$LOG" 2>&1; then
    echo "$(date -Is) relay_sync: pushed new relay $URL" >> "$LOG"
else
    echo "$(date -Is) relay_sync: PUSH FAILED for $URL" >> "$LOG"
fi

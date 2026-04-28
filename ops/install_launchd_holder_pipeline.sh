#!/bin/zsh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${LABEL:-com.sharpoc.strategyresearch.holder.pipeline}"
HOUR="${HOUR:-21}"
MINUTE="${MINUTE:-30}"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$LABEL.plist"
SCRIPT_PATH="$REPO_ROOT/ops/run_mac_mini_holder_pipeline.sh"

mkdir -p "$PLIST_DIR" "$REPO_ROOT/ops/logs"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$SCRIPT_PATH</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>ENV_FILE</key>
    <string>$REPO_ROOT/.env.mac_mini</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
  </array>
  <key>StandardOutPath</key>
  <string>$REPO_ROOT/ops/logs/launchd.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$REPO_ROOT/ops/logs/launchd.stderr.log</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
launchctl enable "gui/$(id -u)/$LABEL"

if ! launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  echo "Failed to load $LABEL"
  exit 1
fi
if launchctl print-disabled "gui/$(id -u)" | grep -q "\"$LABEL\" => disabled"; then
  echo "Failed to enable $LABEL"
  exit 1
fi

echo "Installed $LABEL"
echo "plist=$PLIST_PATH"
echo "schedule=weekday ${HOUR}:${MINUTE}"
echo "status=enabled"
echo "manual run: launchctl kickstart -k gui/$(id -u)/$LABEL"

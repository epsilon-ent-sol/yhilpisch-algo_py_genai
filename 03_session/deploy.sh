#!/usr/bin/env bash
set -euo pipefail

# deploy.sh: local deploy or remote install helper
if [[ "${1-}" == "--remote" ]]; then
    # remote installation steps
    echo "=== Remote setup on $(hostname) ==="
    # ensure we run inside the deployment folder
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
    cd "$SCRIPT_DIR"
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip git
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install pandas numpy python-dateutil v20
    pip install git+https://github.com/yhilpisch/tpqoa.git
    echo "Remote setup complete."
    exit 0
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 user@host"
    exit 1
fi

SERVER=$1
REMOTE_DIR=~/mr_trading
echo "Copying scripts and credentials to $SERVER:$REMOTE_DIR ..."
ssh "$SERVER" "mkdir -p $REMOTE_DIR"
scp \
    mr_trading.py tpqoa.py oanda.cfg oanda_instruments.json deploy.sh \
    "$SERVER":$REMOTE_DIR/

echo "Running remote setup..."
ssh "$SERVER" "bash $REMOTE_DIR/deploy.sh --remote"

read -p "Start live trading on $SERVER now? [y/N] " resp
if [[ "$resp" =~ ^[Yy] ]]; then
    ssh "$SERVER" "cd $REMOTE_DIR && source venv/bin/activate && python mr_trading.py --config oanda.cfg --instrument DE30_EUR"
else
    echo "To start later on $SERVER, run:"
    echo "  ssh $SERVER"
    echo "  cd $REMOTE_DIR && source venv/bin/activate && python mr_trading.py --config oanda.cfg --instrument DE30_EUR"
fi

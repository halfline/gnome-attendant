#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting GNOME Attendant Session${NC}"

# Prompt for username if not provided as argument
if [ -z "$1" ]; then
    echo -e "${YELLOW}Enter username for GNOME Attendant user (default: gnome-attendant):${NC}"
    read -p "Username: " USERNAME
    USERNAME=${USERNAME:-gnome-attendant}
else
    USERNAME="$1"
fi

# Validate username
if [[ ! "$USERNAME" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo -e "${RED}❌ Invalid username. Use only lowercase letters, numbers, underscore, and dash${NC}"
    exit 1
fi

# Check if user exists
if ! id "$USERNAME" &>/dev/null; then
    echo -e "${RED}❌ User $USERNAME does not exist${NC}"
    echo -e "${YELLOW}💡 Run setup-session.sh first to create the user and configure RDP${NC}"
    exit 1
fi

# Start the GNOME headless session
echo -e "${BLUE}🖥️  Starting GNOME headless session for $USERNAME...${NC}"
if systemctl start "gnome-headless-session@$USERNAME.service"; then
    echo -e "${GREEN}✅ GNOME headless session started successfully${NC}"
    
    # Show session status
    echo -e "\n${BLUE}📊 Session Status:${NC}"
    systemctl status "gnome-headless-session@$USERNAME.service" --no-pager -l
    
    # Get the RDP port if possible
    RDP_PORT=$(sudo -u "$USERNAME" bash -c "
    export XDG_RUNTIME_DIR=/run/user/\$(id -u)
    export DBUS_SESSION_BUS_ADDRESS=unix:path=\$XDG_RUNTIME_DIR/bus
    grdctl status 2>/dev/null | grep -i 'rdp.*port' | grep -oE '[0-9]+' | head -1
    " 2>/dev/null || echo "3389")
    
    # Default to 3389 if we couldn't detect the port
    RDP_PORT=${RDP_PORT:-3389}
    
    echo -e "\n${GREEN}🎉 Session is now ready for RDP connections!${NC}"
    echo -e "${BLUE}Connect to:${NC} localhost:$RDP_PORT"
    echo -e "${BLUE}Username:${NC} $USERNAME"
else
    echo -e "${RED}❌ Failed to start GNOME headless session${NC}"
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo -e "   1. Ensure setup-session.sh was run successfully"
    echo -e "   2. Check if the $USERNAME user exists: id $USERNAME"
    echo -e "   3. Check service logs: journalctl -u gnome-headless-session@$USERNAME.service"
    exit 1
fi

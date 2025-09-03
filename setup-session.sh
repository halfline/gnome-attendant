#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 GNOME Attendant Session Setup${NC}"
echo -e "Setting up GNOME Attendant user and RDP configuration...\n"

# Check if running with sufficient privileges and silently escalate if needed
if [ "$EUID" -ne 0 ]; then
    if command -v pkexec >/dev/null 2>&1; then
        pkexec "$0" "$@"
        exit $?
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$0" "$@"
        exit $?
    else
        echo -e "${RED}❌ Neither pkexec nor sudo available${NC}"
        echo -e "${YELLOW}   Please run as root or install pkexec/sudo${NC}"
        exit 1
    fi
fi

# Prompt for username
echo -e "${YELLOW}Enter username for GNOME Attendant user (default: gnome-attendant):${NC}"
read -p "Username: " USERNAME
USERNAME=${USERNAME:-gnome-attendant}

# Validate username
if [[ ! "$USERNAME" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo -e "${RED}❌ Invalid username. Use only lowercase letters, numbers, underscore, and dash${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Using username: $USERNAME${NC}\n"

# Prompt for password securely
echo -e "${YELLOW}Please enter a password for the $USERNAME user:${NC}"
read -s -p "Password: " PASSWORD
echo
read -s -p "Confirm password: " PASSWORD_CONFIRM
echo

# Verify passwords match
if [ "$PASSWORD" != "$PASSWORD_CONFIRM" ]; then
    echo -e "${RED}❌ Passwords do not match. Exiting.${NC}"
    exit 1
fi

if [ -z "$PASSWORD" ]; then
    echo -e "${RED}❌ Password cannot be empty. Exiting.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Passwords match${NC}\n"

# Create user account
echo -e "${BLUE}📝 Creating $USERNAME user...${NC}"
useradd -m "$USERNAME" 2>/dev/null || {
    if id "$USERNAME" &>/dev/null; then
        echo -e "${YELLOW}⚠️  User $USERNAME already exists${NC}"
    else
        echo -e "${RED}❌ Failed to create user $USERNAME${NC}"
        exit 1
    fi
}

# Set password securely using chpasswd
echo -e "${BLUE}🔑 Setting user password...${NC}"
echo "$USERNAME:$PASSWORD" | chpasswd
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Password set successfully${NC}"
else
    echo -e "${RED}❌ Failed to set password${NC}"
    exit 1
fi

# Setup directories and certificates
echo -e "${BLUE}📁 Setting up directories and certificates...${NC}"
USER_HOME="$(eval realpath ~$USERNAME)"
DIR="$USER_HOME/.local/share/gnome-remote-desktop"
mkdir -p "$DIR"

# Generate RDP certificates with proper error handling
echo -e "${BLUE}🔐 Generating SSL certificates for RDP...${NC}"
if openssl req -new -newkey rsa:4096 -days 720 -nodes -x509 \
    -subj "/C=US/ST=Local/L=Local/O=GNOME-Attendant/CN=localhost" \
    -out "$DIR/rdp.crt" \
    -keyout "$DIR/rdp.key"; then
    echo -e "${GREEN}✅ SSL certificates generated successfully${NC}"
else
    echo -e "${RED}❌ Failed to generate SSL certificates${NC}"
    exit 1
fi

# Verify certificates were created and are valid
if [ -f "$DIR/rdp.crt" ] && [ -f "$DIR/rdp.key" ] && [ -s "$DIR/rdp.crt" ] && [ -s "$DIR/rdp.key" ]; then
    # Test certificate validity
    if openssl x509 -in "$DIR/rdp.crt" -noout -text >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Certificates validated${NC}"
    else
        echo -e "${YELLOW}⚠️  Certificate validation warning - but continuing${NC}"
    fi
else
    echo -e "${RED}❌ Certificate files are missing or empty${NC}"
    exit 1
fi

chown "$USERNAME:$USERNAME" -R "$USER_HOME"

# Start the headless session
echo -e "${BLUE}🚀 Starting GNOME headless session...${NC}"
systemctl start "gnome-headless-session@$USERNAME.service"

# Configure GNOME Remote Desktop
echo -e "${BLUE}🖥️  Configuring GNOME Remote Desktop...${NC}"
sudo -u "$USERNAME" bash -c "
sleep 2
export XDG_RUNTIME_DIR=/run/user/\$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=\$XDG_RUNTIME_DIR/bus

# Set TLS certificates
grdctl --headless rdp set-tls-cert ~/.local/share/gnome-remote-desktop/rdp.crt
grdctl --headless rdp set-tls-key ~/.local/share/gnome-remote-desktop/rdp.key

# Set RDP credentials - try different methods for compatibility
if echo '$PASSWORD' | grdctl --headless rdp set-credentials '$USERNAME' --password 2>/dev/null; then
    echo 'RDP credentials set via --password flag'
elif printf '%s\n' '$PASSWORD' '$PASSWORD' | grdctl --headless rdp set-credentials '$USERNAME' 2>/dev/null; then
    echo 'RDP credentials set via stdin'
else
    # If automated methods fail, we'll handle this in the error check below
    grdctl --headless rdp set-credentials '$USERNAME' || true
fi

# Enable RDP
grdctl --headless rdp enable

# Enable the headless service
systemctl --user enable --now gnome-remote-desktop-headless.service
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ GNOME Remote Desktop configured${NC}"
else
    echo -e "${YELLOW}⚠️  GNOME Remote Desktop configuration may have failed${NC}"
    echo -e "${YELLOW}    You may need to set credentials manually with:${NC}"
    echo -e "${YELLOW}    sudo -u $USERNAME grdctl --headless rdp set-credentials $USERNAME${NC}"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ GNOME headless session started${NC}"
else
    echo -e "${YELLOW}⚠️  Failed to start session automatically${NC}"
    echo -e "${YELLOW}    You can start it manually with:${NC}"
    echo -e "${YELLOW}    systemctl start gnome-headless-session@$USERNAME.service${NC}"
fi

# Get the actual RDP port from grdctl status
echo -e "${BLUE}🔍 Detecting RDP port...${NC}"
RDP_PORT=$(sudo -u "$USERNAME" bash -c "
export XDG_RUNTIME_DIR=/run/user/\$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=\$XDG_RUNTIME_DIR/bus
grdctl status 2>/dev/null | grep -i 'rdp.*port' | grep -oE '[0-9]+' | head -1
" 2>/dev/null || echo "3389")

# Default to 3389 if we couldn't detect the port
RDP_PORT=${RDP_PORT:-3389}

echo -e "\n${GREEN}🎉 Setup completed!${NC}"
echo -e "${BLUE}User:${NC} $USERNAME"
echo -e "${BLUE}RDP Port:${NC} $RDP_PORT"
echo -e "${BLUE}Certificates:${NC} $DIR"

# Now configure GNOME Attendant with the connection details
echo -e "\n${BLUE}🤖 Configuring GNOME Attendant...${NC}"

SCRIPT_DIR="$(realpath $(dirname "$0"))"
GNOME_ATTENDANT="$SCRIPT_DIR/gnome-attendant"

echo -e "${BLUE}📝 Pre-configuring GNOME Attendant connection settings...${NC}"

# Create a temporary configuration script
CONFIG_SCRIPT=$(mktemp)
cat > "$CONFIG_SCRIPT" << EOF
/host localhost:$RDP_PORT
/username $USERNAME
/password $PASSWORD
EOF

if "$GNOME_ATTENDANT" < "$CONFIG_SCRIPT"; then
    echo -e "${GREEN}✅ GNOME Attendant configured successfully${NC}"
    echo -e "${BLUE}Connection details saved. You can now run:${NC}"
    echo -e "${YELLOW}    ./gnome-attendant${NC}"
    echo -e "${BLUE}And it will auto-connect using the saved credentials.${NC}"
else
    echo -e "${YELLOW}⚠️  Could not auto-configure GNOME Attendant${NC}"
    echo -e "${BLUE}You can manually configure it by running:${NC}"
    echo -e "${YELLOW}    ./gnome-attendant${NC}"
    echo -e "${BLUE}Then use these commands:${NC}"
    echo -e "${YELLOW}    /host localhost:$RDP_PORT${NC}"
    echo -e "${YELLOW}    /username $USERNAME${NC}"
    echo -e "${YELLOW}    /password${NC}"
fi

# Clean up temporary file
rm -f "$CONFIG_SCRIPT"

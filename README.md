# GNOME Attendant

**AI-Powered Desktop Automation for GNOME Environments**

GNOME Attendant is an intelligent desktop automation tool that leverages AI to interact with and control GNOME desktop environments. It provides automated assistance for various desktop tasks and workflows.

## Features

- 🤖 **AI-Powered Automation** - Intelligent desktop interaction using advanced AI models
- 🖥️ **GNOME Integration** - Native support for GNOME desktop environments
- 🔗 **RDP Support** - Remote desktop protocol functionality via FreeRDP
- 📦 **Zero Installation** - Uses `uvx` for dependency management (no pip install needed)
- 🎨 **Rich Terminal Output** - Beautiful command-line interface with rich formatting

## Prerequisites

- Python 3.8+
- `uvx` (Python package runner)
- GNOME desktop environment
- FreeRDP (for remote desktop functionality)

## Installation

### Install uvx (if not already installed)
```bash
pip install --user uvx
```

### Clone and Setup
```bash
git clone <repository-url>
cd gnome-attendant
chmod +x gnome-attendant
```

## Usage

### Basic Usage
```bash
./gnome-attendant [options]
```

The launcher script automatically handles all Python dependencies via `uvx`, so no manual package installation is required.

### Session Management

The session management scripts help set up and manage headless sessions:

```bash
# Setup a new session (creates user, configures RDP settings)
./setup-session.sh

# Start a session (launches headless session)
./start-session.sh
```

The `setup-session.sh` script performs one-time configuration:
- Creates a dedicated user account for remote access
- Configures RDP settings and permissions
- Sets up necessary system services

The `start-session.sh` script starts the headless GNOME session.
- Enables remote access to the current session

Both scripts will prompt for authentication when required. The session remains active until explicitly stopped or the system is shut down.

### Interactive Mode
When run without arguments, GNOME Attendant starts in interactive mode:
```bash
./gnome-attendant
```

In interactive mode, you can:
- Type natural language commands directly
- Use slash commands for specific actions
- Get real-time feedback and suggestions
- Control multiple sessions
- View command history

Example interactive session:
```
> open firefox
Opening Firefox browser...

> maximize window
Window maximized

> /connect
Connecting to remote session...
Connected successfully

> /help
Available commands:
  /connect    - Connect to a remote session
  /type      - Type text
  /key       - Send keyboard input
  /wait      - Wait for specified seconds
  /import    - Run commands from file
  ...
```

### Script Import
```bash
# Run commands from a file
/import hello.attendant

# Or use in batch mode
echo "/import my-automation.attendant" | ./gnome-attendant
```

## Dependencies

The following Python packages are automatically managed via `uvx`:

- **Pillow** - Image processing and manipulation
- **Anthropic** - AI model integration
- **python-dotenv** - Environment variable management
- **Rich** - Terminal formatting and display
- **platformdirs** - XDG Base Directory specification compliance
- **keyring** - Secure credential storage using system keyring
- **dwarfbind** - Dynamic library binding generation

## Project Structure

```
gnome-attendant/
├── gnome-attendant          # Main launcher script
├── gnome-attendant.py       # Core Python application
├── freerdp.py              # RDP functionality
├── setup-session.sh        # Session setup script
├── start-session.sh        # Session start script
├── hello.attendant         # Example automation script
├── scripts/                # Internal tools and utilities
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Script Import Format

The `/import` command supports script files with the following format:

```bash
# Comments start with # and are ignored
# Empty lines are also ignored

# Connection commands
/connect

# Natural language commands
click on Activities
open Settings
open Terminal

# Specific automation commands
/type "Hello World!"
/key Return
/wait 2

# More natural language
close the application
```

## Configuration

Create a `.env` file in the project directory for environment-specific configuration:

```bash
# Example .env file
API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## Development

### Running from Source
```bash
python gnome-attendant.py [arguments]
```

### Project Configuration
The project uses `pyproject.toml` for configuration management.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the project repository.

## Acknowledgments

- Built with the Anthropic AI API
- Uses uvx for seamless Python dependency management
- Integrates with GNOME desktop environment
- Uses dwarfbind for dynamic library bindings 
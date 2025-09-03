#!/usr/bin/env python3
"""
GNOME Attendant - AI-Powered Desktop Automation

A sophisticated RDP automation tool leveraging LLMs that can interact with
remote desktop sessions through intelligent visual analysis and automated actions.

Features:
- RDP connectivity using FreeRDP3
- AI-powered visual analysis via Claude
- Natural language command processing
- Screenshot-based desktop interaction
- Automated mouse and keyboard control

Usage:
    ./gnome-attendant                    # Interactive mode
    echo "command" | ./gnome-attendant   # Batch mode
    ./gnome-attendant --help            # Show help

Dependencies:
    - FreeRDP3 libraries (freerdp-devel)
    - Python packages: pillow, anthropic, python-dotenv, rich, platformdirs, keyring
    - Valid Anthropic API key

Example:
    ./gnome-attendant
    > open the Activities overview
    > search for "Text Editor" and press Enter
    > type hello world

Author: Ray Strode
License: MIT
"""

# Standard library imports
import base64
import gc
import getpass
import json
import logging
import os
import re
import readline
import shlex
import sys
import time
import subprocess
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

# Third-party imports
import anthropic
from dotenv import load_dotenv
import keyring
from PIL import Image
from platformdirs import user_config_dir, user_data_dir
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Local imports
from freerdp import FreeRDPClient

# Initialize environment
load_dotenv()

# Configure logging
logging.basicConfig(
	level=logging.WARNING,  # Default to WARNING to avoid exposing sensitive info
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class Attendant:
	"""
	Main automation class for GNOME Attendant.

	Provides RDP-based desktop automation using Claude AI for intelligent action 
	planning. Supports both interactive REPL mode and batch processing of 
	automation tasks.

	The attendant maintains state about the RDP connection, handles user input,
	manages configuration, and coordinates between the RDP client and AI for
	executing automation tasks.

	Attributes:
		rdp_host: RDP server hostname
		rdp_port: RDP server port
		rdp_username: RDP authentication username 
		rdp_password: RDP authentication password
		rdp_domain: RDP authentication domain
		connected: Current RDP connection status
		auto_connect: Whether to automatically connect when needed
		freerdp_client: FreeRDP client instance
		anthropic_client: Claude API client instance
		last_screenshot: Most recent screenshot
		action_history: Simple action history for legacy compatibility
		detailed_action_history: Detailed action tracking with success/failure
		console: Rich console for formatted output
	"""

	CONFIG_FILE = os.path.join(user_config_dir("gnome-attendant"), "config.json")

	def __init__(self) -> None:
		"""Initialize the attendant with default configuration."""
		self.console = Console()

		# Ensure config directory exists
		config_dir = os.path.dirname(self.CONFIG_FILE)
		os.makedirs(config_dir, exist_ok=True)

		# Initialize connection settings
		self.rdp_host: Optional[str] = None
		self.rdp_port: Optional[int] = None
		self.rdp_username: Optional[str] = None
		self.rdp_password: Optional[str] = None
		self.rdp_domain: Optional[str] = None
		self.connected: bool = False
		self.auto_connect: bool = True
		
		# Initialize operational settings
		self.connection_check_interval: int = 30  # seconds
		self.last_successful_operation: float = 0
		self.context_history: List[str] = []
		self.session_start_time: datetime = datetime.now()
		self.max_steps: int = 25

		# Initialize clients
		self.freerdp_client: Optional[FreeRDPClient] = None
		self.anthropic_client = self._initialize_anthropic_client()

		# Initialize action tracking
		self.action_history: List[str] = []
		self.detailed_action_history: List[Dict[str, Any]] = []
		self.last_screenshot: Optional[Image.Image] = None

		# Coordinate/screenshot scaling (uniform factor). 1.0 = no scaling
		self.screenshot_scale_factor: float = 1.0
		self.screenshot_max_width: Optional[int] = None
		self.scale_strategy: str = 'factor'  # 'factor' or 'max_width'
		self.last_effective_scale_factor: float = 1.0

		# Load persisted configuration
		self.load_config()

		# Try to auto-connect if configured
		if self.auto_connect and self.rdp_host and self.rdp_password:
			logger.debug("Auto-connect enabled and credentials available")

	def _initialize_anthropic_client(self) -> Optional[anthropic.Anthropic]:
		"""
		Initialize the Anthropic API client.

		Returns:
			Optional[anthropic.Anthropic]: Configured client or None if setup fails
		"""
		api_key = self._get_api_key()
		if not api_key:
			self.console.print(
				"⚠️  Warning: API key not set. Use '/api_key <your_key>' or set "
				"ANTHROPIC_API_KEY. AI features disabled.", 
				style="yellow"
			)
			return None

		try:
			return anthropic.Anthropic(api_key=api_key)
		except Exception as error:
			self.console.print(
				f"⚠️  Warning: Failed to initialize Anthropic client: {error}",
				style="yellow"
			)
			return None

	def _get_api_key(self):
		"""
		Get API key from keyring or environment variable.

		Returns:
			str or None: API key if found, None otherwise
		"""
		try:
			# Try to get from keyring first
			api_key = keyring.get_password("gnome-attendant", "anthropic-api-key")
			if api_key:
				return api_key
		except Exception as e:
			logger.debug(f"Failed to access keyring: {e}")

		# Fall back to environment variable
		return os.getenv('ANTHROPIC_API_KEY')

	def _set_api_key(self, api_key):
		"""
		Store API key in keyring.

		Args:
			api_key (str): API key to store

		Returns:
			bool: True if successful, False otherwise
		"""
		try:
			keyring.set_password("gnome-attendant", "anthropic-api-key", api_key)
			return True
		except Exception as e:
			logger.debug(f"Failed to store API key in keyring: {e}")
			return False



	def load_config(self):
		"""Load configuration from file."""
		try:
			if os.path.exists(self.CONFIG_FILE):
				with open(self.CONFIG_FILE, 'r') as f:
					config = json.load(f)
					self.rdp_host = config.get('rdp_host', 'localhost')
					self.rdp_port = config.get('rdp_port', 3389)
					self.rdp_username = config.get('rdp_username')
					self.rdp_password = config.get('rdp_password')
					self.rdp_domain = config.get('rdp_domain')
					self.screenshot_scale_factor = float(config.get('screenshot_scale_factor', 1.0))
					max_width_value = config.get('screenshot_max_width')
					self.screenshot_max_width = int(max_width_value) if max_width_value is not None else None
					self.scale_strategy = config.get('scale_strategy', 'factor')
					fbw = config.get('framebuffer_width')
					fbh = config.get('framebuffer_height')
					self.framebuffer_width = int(fbw) if fbw is not None else None
					self.framebuffer_height = int(fbh) if fbh is not None else None
					self.framebuffer_follow_server = bool(config.get('framebuffer_follow_server', True))
					logger.debug(f"Loaded configuration from {self.CONFIG_FILE}")
			else:
				# Set defaults
				self.rdp_host = 'localhost'
				self.rdp_port = 3389
				self.rdp_username = None
				self.rdp_password = None
				self.rdp_domain = None
				self.screenshot_scale_factor = 1.0
				self.screenshot_max_width = None
				self.scale_strategy = 'factor'
				self.framebuffer_width = None
				self.framebuffer_height = None
				self.framebuffer_follow_server = True
				logger.debug("Using default configuration")

		except Exception as e:
			logger.warning(f"Error loading config: {e}")
			self.rdp_host = 'localhost'
			self.rdp_port = 3389
			self.rdp_username = None
			self.rdp_password = None
			self.rdp_domain = None
			self.screenshot_scale_factor = 1.0
			self.screenshot_max_width = None
			self.scale_strategy = 'factor'
			self.framebuffer_width = None
			self.framebuffer_height = None
			self.framebuffer_follow_server = True

	def save_config(self):
		"""Save configuration to file with proper directory creation."""
		try:
			# Ensure the config directory exists
			config_dir = os.path.dirname(self.CONFIG_FILE)
			os.makedirs(config_dir, exist_ok=True)

			config_data = {
				'rdp_host': self.rdp_host,
				'rdp_port': self.rdp_port,
				'rdp_username': self.rdp_username,
				'rdp_password': self.rdp_password,
				'rdp_domain': self.rdp_domain,
				'screenshot_scale_factor': self.screenshot_scale_factor,
				'screenshot_max_width': self.screenshot_max_width,
				'scale_strategy': self.scale_strategy,
				'framebuffer_width': self.framebuffer_width,
				'framebuffer_height': self.framebuffer_height,
				'framebuffer_follow_server': self.framebuffer_follow_server,
			}

			with open(self.CONFIG_FILE, 'w') as f:
				json.dump(config_data, f, indent=2)
			logger.debug(f"Configuration saved to {self.CONFIG_FILE}")
			return True
		except Exception as e:
			logger.error(f"Error saving config: {e}")
			return False

	def add_action_to_history(self, action_type, params, success, error_msg=None):
		"""
		Add an executed action to both simple and detailed history tracking.

		Args:
			action_type (str): Type of action performed (click, type, key, etc.)
			params: Action parameters (coordinates, text, key name, etc.)
			success (bool): Whether the action completed successfully
			error_msg (str, optional): Error message if action failed
		"""
		action_desc = f"{action_type}:{params}" if params else action_type
		self.action_history.append(action_desc)

		detailed_entry = {
			'action': action_desc,
			'success': success,
			'timestamp': time.time()
		}
		if error_msg:
			detailed_entry['error'] = error_msg

		self.detailed_action_history.append(detailed_entry)

		if len(self.detailed_action_history) > 20:
			self.detailed_action_history = self.detailed_action_history[-20:]

	def get_action_context_for_claude(self):
		"""
		Generate formatted action history context for Claude AI analysis.

		Returns:
			str: Formatted string containing recent action history with success/failure status
		"""
		if not self.detailed_action_history:
			return "No previous actions in this session."

		context_parts = []
		recent_actions = self.detailed_action_history[-10:]

		for i, entry in enumerate(recent_actions, 1):
			status = "✓ SUCCESS" if entry['success'] else "✗ FAILED"
			action_line = f"{i}. {entry['action']} → {status}"
			if not entry['success'] and 'error' in entry:
				action_line += f" (Error: {entry['error']})"
			context_parts.append(action_line)

		return "Previous actions in this session:\n" + "\n".join(context_parts)

	def connect_rdp(self) -> bool:
		"""
		Establish connection to RDP server using FreeRDP3.

		Handles connection setup, authentication, and error reporting. Maintains
		connection state and updates last successful operation time on success.

		Returns:
			bool: True if connection successful, False otherwise
		"""
		try:
			logger.info(f"Attempting to connect to RDP at {self.rdp_host}:{self.rdp_port}")

			if not self.rdp_host:
				logger.error("No RDP host configured. Use /host to set the server.")
				return False

			self.freerdp_client = FreeRDPClient()

			# Force a specific remote desktop size
			requested_width, requested_height = 1920, 1080

			success = self.freerdp_client.connect(
				hostname=self.rdp_host,
				port=int(self.rdp_port or 3389),
				username=self.rdp_username or "",
				password=self.rdp_password or "",
				domain=self.rdp_domain or "",
				width=int(requested_width),
				height=int(requested_height)
			)

			if success:
				logger.info(f"Connected to RDP server at {self.rdp_host}:{self.rdp_port}")
				self.connected = True
				self.last_successful_operation = time.time()
				# If following server resolution, capture negotiated size and persist
				try:
					if self.framebuffer_follow_server and hasattr(self.freerdp_client, 'get_framebuffer_size'):
						negotiated_width, negotiated_height = self.freerdp_client.get_framebuffer_size()
						if negotiated_width and negotiated_height:
							self.framebuffer_width = int(negotiated_width)
							self.framebuffer_height = int(negotiated_height)
							self.save_config()
				except Exception as negotiated_error:
					logger.debug(f"Could not persist negotiated framebuffer size: {negotiated_error}")
				return True
			else:
				logger.warning(
					"FreeRDP connection failed. Check credentials and server status."
				)
				return False

		except ValueError as error:
			logger.error(f"Invalid connection parameters: {error}")
			return False
		except Exception as error:
			logger.error(f"Failed to connect to RDP: {error}")
			return False

	def disconnect_rdp(self) -> None:
		"""
		Cleanly disconnect from RDP server and perform cleanup.

		Ensures proper resource cleanup and state reset, including:
		- Disconnecting the FreeRDP client
		- Resetting connection state
		- Running garbage collection to clean up resources
		"""
		if self.freerdp_client and self.connected:
			try:
				self.freerdp_client.disconnect()
				self.console.print("Disconnected from RDP server", style="yellow")
			except Exception as error:
				logger.debug(f"Error during RDP disconnect: {error}")
			finally:
				self.freerdp_client = None
				self.connected = False

		gc.collect()

	def check_connection(self) -> bool:
		"""
		Verify RDP connection health and reconnect if necessary.

		Implements intelligent connection checking with time-based optimization
		to avoid excessive connection tests. Only performs full check after
		connection_check_interval has elapsed since last successful operation.

		Returns:
			bool: True if connection is healthy, False if reconnection failed
		"""
		try:
			if not self.freerdp_client:
				logger.debug("RDP client is None, attempting to reconnect...")
				return self.connect_rdp()

			current_time = time.time()
			if (current_time - self.last_successful_operation 
				< self.connection_check_interval):
				return True

			if (hasattr(self.freerdp_client, 'is_connected') 
				and not self.freerdp_client.is_connected()):
				logger.debug("RDP client shows disconnected status, reconnecting...")
				return self.connect_rdp()

			try:
				# Simple attribute access to test connection
				_ = self.freerdp_client.x if hasattr(self.freerdp_client, 'x') else 0
				self.last_successful_operation = current_time
				return True
			except Exception as test_error:
				logger.debug(f"Connection test failed: {test_error}, reconnecting...")
				return self.connect_rdp()

		except Exception as error:
			logger.debug(f"Connection check failed: {error}, attempting to reconnect...")
			return self.connect_rdp()

	def auto_connect_if_needed(self) -> bool:
		"""
		Automatically establish RDP connection if not currently connected.

		Provides a convenient way to ensure connection is available before
		performing operations. Handles user feedback and error reporting.

		Returns:
			bool: True if connected (either already or newly), False if failed
		"""
		if not self.connected:
			try:
				self.console.print("🔌 Auto-connecting to RDP server...", 
								 style="yellow")
				with self.console.status("🔌 Connecting..."):
					if self.connect_rdp():
						self.connected = True
						self.console.print("✅ Auto-connected successfully!", 
										 style="green")
						return True
					else:
						self.console.print("❌ Auto-connection failed", style="red")
						self.console.print(
							"💡 Try '/connect' for manual connection or "
							"'/reconnect' to restart process", 
							style="dim"
						)
						return False
			except KeyboardInterrupt:
				self.console.print("\n⏹️  Auto-connection cancelled", 
								 style="yellow")
				return False
		return True

	def take_screenshot(self, silent: bool = False) -> Optional[Image.Image]:
		"""
		Capture a screenshot from the RDP session using FreeRDP3.

		Handles connection verification, error cases, and provides fallback
		behavior when screenshot capture fails.

		Args:
			silent: If True, suppress debug logging output

		Returns:
			PIL.Image: Screenshot image object or None if capture fails

		Raises:
			ValueError: If RDP client is not connected or screenshot fails
		"""
		if not self.connected or not self.freerdp_client:
			raise ValueError("RDP client not connected")

		if not self.check_connection():
			raise ValueError("Cannot establish RDP connection for screenshot")

		try:
			if not silent:
				logger.debug("Taking screenshot via FreeRDP...")

			screenshot_capture = self.freerdp_client.get_screenshot()

			if screenshot_capture:
				# Convert ScreenCapture RGB data to PIL Image
				image = Image.frombytes(
					'RGB', 
					(screenshot_capture.width, screenshot_capture.height),
					screenshot_capture.data
				)

				if not silent:
					logger.debug(
						f"Screenshot captured successfully: {image.size}"
					)
				self.last_successful_operation = time.time()
				return image
			else:
				# Create a placeholder screenshot on failure
				if not silent:
					logger.warning(
						"No screenshot data available, creating placeholder"
					)
				return Image.new('RGB', (1024, 768), color='darkblue')

		except Exception as error:
			if not silent:
				logger.debug(f"Screenshot failed: {error}")
			try:
				# Create fallback placeholder
				placeholder = Image.new('RGB', (1024, 768), color='darkblue')
				logger.warning("Screenshot failed, returning placeholder image")
				return placeholder
			except Exception as placeholder_error:
				raise ValueError(
					f"Screenshot capture failed and could not create placeholder: "
					f"{error}. Placeholder error: {placeholder_error}"
				) from error

	def _create_screenshot_from_update(self, update_data):
		"""
		Create a PIL Image from RDP screen update data.

		Args:
			update_data (dict): Screen update data from rdpy3

		Returns:
			PIL.Image: Screenshot image or None if failed
		"""
		try:
			width = update_data['width']
			height = update_data['height']
			bits_per_pixel = update_data['bitsPerPixel']
			data = update_data['data']

			if bits_per_pixel == 32:
				# 32-bit RGBA
				mode = 'RGBA'
				raw_mode = 'RGBA'
			elif bits_per_pixel == 24:
				# 24-bit RGB
				mode = 'RGB'
				raw_mode = 'RGB'
			elif bits_per_pixel == 16:
				# 16-bit RGB565
				mode = 'RGB'
				raw_mode = 'RGB;16'
			else:
				logger.warning(f"Unsupported bits per pixel: {bits_per_pixel}")
				return None

			# Create image from raw data
			image = Image.frombytes(mode, (width, height), data, 'raw', raw_mode)

			return image

		except Exception as e:
			logger.error(f"Error creating screenshot from update data: {e}")
			return None

	def image_to_base64(self, image):
		"""
		Convert PIL Image to base64 encoded string for API transmission.

		Args:
			image (PIL.Image): Image to convert

		Returns:
			str: Base64 encoded PNG image data
		"""
		buffer = BytesIO()
		image.save(buffer, format='PNG')
		img_data = buffer.getvalue()
		return base64.b64encode(img_data).decode()

	def _resolve_key_code(self, key_name):
		"""
		Resolve a key name to its X11 keysym code for precise RDP input.

		Provides comprehensive mapping for:
		- Single characters (a-z, 0-9, punctuation)
		- Function keys (F1-F12)
		- Special keys (Return, Tab, Escape, etc.)
		- Numeric keypad keys
		- Arrow keys
		- Modifier keys
		- Media keys

		Args:
			key_name (str): Name of the key to resolve

		Returns:
			int or None: X11 keysym code for the key, or None if not found
		"""
		key_lower = key_name.lower()

		if len(key_name) == 1:
			char = key_name.lower()
			if 'a' <= char <= 'z':
				return ord(char)
			elif '0' <= char <= '9':
				return ord(char)
			else:
				punctuation_map = {
					' ': 32,
					'!': 33, '"': 34, '#': 35, '$': 36, '%': 37, '&': 38, "'": 39,
					'(': 40, ')': 41, '*': 42, '+': 43, ',': 44, '-': 45, '.': 46, '/': 47,
					':': 58, ';': 59, '<': 60, '=': 61, '>': 62, '?': 63, '@': 64,
					'[': 91, '\\': 92, ']': 93, '^': 94, '_': 95, '`': 96,
					'{': 123, '|': 124, '}': 125, '~': 126
				}
				return punctuation_map.get(char)

		key_map = {
			'f1': 65470, 'f2': 65471, 'f3': 65472, 'f4': 65473,
			'f5': 65474, 'f6': 65475, 'f7': 65476, 'f8': 65477,
			'f9': 65478, 'f10': 65479, 'f11': 65480, 'f12': 65481,

			'return': 65293, 'enter': 65293,
			'tab': 65289,
			'escape': 65307, 'esc': 65307,
			'space': 32,
			'backspace': 65288,
			'delete': 65535,
			'insert': 65379,
			'home': 65360,
			'end': 65367,
			'page_up': 65365, 'pageup': 65365,
			'page_down': 65366, 'pagedown': 65366,

			'up': 65362, 'down': 65364, 'left': 65361, 'right': 65363,
			'arrow_up': 65362, 'arrow_down': 65364,
			'arrow_left': 65361, 'arrow_right': 65363,

			'shift': 65505, 'shift_l': 65505, 'shift_r': 65506,
			'ctrl': 65507, 'control': 65507, 'ctrl_l': 65507, 'ctrl_r': 65508,
			'alt': 65513, 'alt_l': 65513, 'alt_r': 65514,
			'super': 65515, 'super_l': 65515, 'super_r': 65516,
			'menu': 65383,

			'kp_0': 65456, 'kp_1': 65457, 'kp_2': 65458, 'kp_3': 65459,
			'kp_4': 65460, 'kp_5': 65461, 'kp_6': 65462, 'kp_7': 65463,
			'kp_8': 65464, 'kp_9': 65465,
			'kp_add': 65451, 'kp_subtract': 65453, 'kp_multiply': 65450,
			'kp_divide': 65455, 'kp_decimal': 65454, 'kp_enter': 65421,

			'caps_lock': 65509, 'capslock': 65509,
			'num_lock': 65407, 'numlock': 65407,
			'scroll_lock': 65300, 'scrolllock': 65300,

			'print': 65377, 'print_screen': 65377,
			'pause': 65299,
			'break': 65387,

			'volume_up': 65469, 'volume_down': 65468, 'volume_mute': 65467,

			'plus': 43, 'minus': 45, 'equals': 61,
			'comma': 44, 'period': 46, 'slash': 47,
			'semicolon': 59, 'apostrophe': 39, 'quote': 39,
			'bracket_left': 91, 'bracket_right': 93,
			'backslash': 92, 'grave': 96, 'tilde': 126,
		}

		if key_lower in key_map:
			return key_map[key_lower]

		if key_lower.startswith('key_') and len(key_lower) == 5:
			digit = key_lower[4]
			if '0' <= digit <= '9':
				return ord(digit)

		func_match = re.match(r'(?:function_?|fn)(\d+)$', key_lower)
		if func_match:
			fn_num = int(func_match.group(1))
			if 1 <= fn_num <= 12:
				return 65470 + fn_num - 1

		return None

	def ask_claude_for_action(self, screenshot, task_description, previous_actions=None):
		"""
		Request next action from Claude AI based on current screenshot and task context.

		Args:
			screenshot (PIL.Image): Current desktop screenshot
			task_description (str): Description of the task to accomplish
			previous_actions (list, optional): Recent actions for context

		Returns:
			tuple: (action_type, parameters) or (None, None) if no action determined
		"""
		# Compute effective scaling for AI context
		original_width, original_height = screenshot.size
		logger.info(f"Original screenshot size: {original_width}x{original_height}")
		image_for_ai = screenshot
		effective_scale_factor = 1.0
		if self.scale_strategy == 'max_width' and self.screenshot_max_width:
			maximum_width = int(self.screenshot_max_width)
			if maximum_width > 0 and original_width > maximum_width:
				new_width = maximum_width
				new_height = max(1, int(round(original_height * (new_width / original_width))))
				effective_scale_factor = new_width / original_width
				try:
					image_for_ai = screenshot.resize((new_width, new_height), Image.LANCZOS)
				except Exception:
					image_for_ai = screenshot.resize((new_width, new_height))
		elif self.scale_strategy == 'factor' and isinstance(self.screenshot_scale_factor, (int, float)) and self.screenshot_scale_factor > 0 and self.screenshot_scale_factor != 1.0:
			new_width = max(1, int(round(original_width * self.screenshot_scale_factor)))
			new_height = max(1, int(round(original_height * self.screenshot_scale_factor)))
			effective_scale_factor = new_width / original_width if original_width else 1.0
			try:
				image_for_ai = screenshot.resize((new_width, new_height), Image.LANCZOS)
			except Exception:
				image_for_ai = screenshot.resize((new_width, new_height))
		self.last_effective_scale_factor = effective_scale_factor
		base64_image = self.image_to_base64(image_for_ai)
		width, height = image_for_ai.size
		# Cross-check with framebuffer size from RDP client (authoritative)
		logger.info(f"LLM prompt resolution: {width}x{height}")
 
		system_prompt = """You are a desktop automation assistant running in a GNOME Shell environment.
 
Current framebuffer resolution: {width}x{height} pixels.
Always output coordinates in this exact space.
 
- You can see the current desktop screenshot and need to determine the next action to complete the given task.
- The activities overview is available by clicking the upper left corner of the screen. From there, a search field is available and a dock is present at the bottom of the screen. If you're unsure what to do, make a best guess and iterate until successful.
- At every step analyze a screenshot and determine the current state of the system and if the task is already completed. Do not assume state based on previously completed actions without verification.
- When identifying windows and applications, look for the window titlebar and the application name.
- Do not confuse the system wallpaper with a running application. If there is no titlebar, it is likely a wallpaper.
- The dock in the activities overview also shows application state by a dot under the application icon.
- If the application is not loaded, there will be no dot under its icon, but there may be an icon.
- If an action doesn't seem to work, it may just need more time to complete. If an application may be slow to load, wait a few seconds before trying alternative approaches to starting it.
- Do not confuse the system panel at the top of the screen with an application titlebar.

Available commands:
1. /click x y - Click at coordinates
2. /double_click x y - Double click at coordinates
3. /right_click x y - Right click at coordinates
4. /type text - Type text
5. /key keyname - Press a key (e.g., 'Return', 'Tab', 'ctrl+c')
6. /wait seconds - Wait for specified seconds
7. /done - Task completed successfully

Respond with ONLY the command in this exact format:

Examples:
/click 100 200
/type hello world
/key Return
/wait 2
/done

- Be precise with coordinates and consider the current state of the desktop.
- Learn from previous failed actions and try different approaches if needed.
- Leave the desktop in a state the user likely wants to see:
  - Don't leave menus opened that are not needed.
  - Exit the activities overview if relevant."""

		action_context = self.get_action_context_for_claude()

		user_message = f"Task: {task_description}\n\n{action_context}\n\nWhat should be the next action based on this screenshot?"

		try:
			response = self.anthropic_client.messages.create(
				model="claude-3-5-sonnet-20241022",
				max_tokens=200,
				system=system_prompt.format(width=width, height=height),
				messages=[
					{
						"role": "user",
						"content": [
							{
								"type": "image",
								"source": {
									"type": "base64",
									"media_type": "image/png",
									"data": base64_image
								}
							},
							{
								"type": "text",
								"text": user_message
							}
						]
					}
				]
			)

			action_text = response.content[0].text.strip()
			print(f"🤖 {action_text}")

			action_lines = action_text.split('\n')
			actual_action = action_lines[-1].strip()

			return self.parse_action(actual_action)

		except Exception as e:
			print(f"Error communicating with Claude: {e}")
			return None, None

	def parse_action(self, action_text):
		"""
		Parse Claude's response text into structured action type and parameters.

		Handles both old format (action_type:parameters) and new format (/command args).

		Args:
			action_text (str): Raw action text from Claude

		Returns:
			tuple: (action_type, parameters) or (None, None) if parsing fails
		"""
		action_text = action_text.strip()

		# Handle new slash command format
		if action_text.startswith('/'):
			try:
				parts = action_text[1:].split(None, 1)  # Split on first whitespace
				command = parts[0].lower()
				args = parts[1] if len(parts) > 1 else ""

				if command == 'done':
					return 'done', None
				elif command in ['click', 'double_click', 'right_click']:
					try:
						coords = args.split()
						if len(coords) >= 2:
							x, y = int(coords[0]), int(coords[1])
							return command, (x, y)
					except ValueError:
						return None, None
				elif command == 'type':
					return 'type', args
				elif command == 'key':
					return 'key', args
				elif command == 'wait':
					try:
						seconds = float(args)
						return 'wait', seconds
					except ValueError:
						return None, None

				return None, None
			except:
				return None, None

		# Handle legacy 'done' response for backward compatibility
		if action_text.lower() == 'done':
			return 'done', None

		# Handle legacy colon format for backward compatibility
		if ':' not in action_text:
			return None, None

		action_type, params = action_text.split(':', 1)
		action_type = action_type.lower().strip()

		if action_type in ['click', 'double_click', 'right_click']:
			try:
				x, y = map(int, params.split(','))
				return action_type, (x, y)
			except ValueError:
				return None, None
		elif action_type == 'type':
			return action_type, params.strip()
		elif action_type == 'key':
			return action_type, params.strip()
		elif action_type == 'wait':
			try:
				seconds = float(params.strip())
				return action_type, seconds
			except ValueError:
				return None, None

		return None, None

	def execute_action(self, action_type, params):
		"""
		Execute a specific automation action via RDP connection using FreeRDP3.

		Args:
			action_type (str): Type of action to execute
			params: Parameters for the action (coordinates, text, key names, etc.)

		Returns:
			bool: True if action executed successfully, False otherwise
		"""
		if not self.connected or not self.freerdp_client:
			raise ValueError("RDP client not connected")

		current_time = time.time()
		if current_time - self.last_successful_operation > self.connection_check_interval:
			if not self.check_connection():
				logger.warning("Failed to establish RDP connection for action execution")
				return False

		try:
			if action_type == 'click':
				x, y = params
				# Transform coordinates back to framebuffer space if scaled
				effective_scale_factor = getattr(self, 'last_effective_scale_factor', 1.0) or 1.0
				if effective_scale_factor != 1.0:
					x = int(round(x / effective_scale_factor))
					y = int(round(y / effective_scale_factor))
				logger.info(f"Executing click at ({x}, {y}) on framebuffer {self.freerdp_client.get_framebuffer_size()}")
				success = self.freerdp_client.click_mouse(x, y, button=1)  # Left click
				if success:
					logger.debug(f"Clicked at ({x}, {y})")
				return success

			elif action_type == 'double_click':
				x, y = params
				if isinstance(self.screenshot_scale_factor, (int, float)) and self.screenshot_scale_factor > 0 and self.screenshot_scale_factor != 1.0:
					x = int(round(x / self.screenshot_scale_factor))
					y = int(round(y / self.screenshot_scale_factor))
				success = self.freerdp_client.click_mouse(x, y, button=1, double_click=True)  # Double click
				if success:
					logger.debug(f"Double-clicked at ({x}, {y})")
				return success

			elif action_type == 'right_click':
				x, y = params
				if isinstance(self.screenshot_scale_factor, (int, float)) and self.screenshot_scale_factor > 0 and self.screenshot_scale_factor != 1.0:
					x = int(round(x / self.screenshot_scale_factor))
					y = int(round(y / self.screenshot_scale_factor))
				success = self.freerdp_client.click_mouse(x, y, button=2)  # Right click
				if success:
					logger.debug(f"Right-clicked at ({x}, {y})")
				return success

			elif action_type == 'type':
				self.freerdp_client.send_text(params)
				logger.debug(f"Typed: {params}")
				return True

			elif action_type == 'key':
				key_name = params.strip()
				# Resolve key name to keycode and send key press + release
				keycode = self._resolve_key_code(key_name)
				if keycode:
					self.freerdp_client.send_key(keycode, True)   # Press
					time.sleep(0.02)
					self.freerdp_client.send_key(keycode, False)  # Release
					logger.debug(f"Sent key: {key_name}")
					return True
				else:
					logger.warning(f"Unknown key: {key_name}")
					return False

			elif action_type == 'wait':
				time.sleep(params)
				logger.debug(f"Waited {params} seconds")
				return True

			# Unknown action type
			logger.warning(f"Unknown action type: {action_type}")
			return False

		except Exception as e:
			logger.error(f"Error executing action {action_type}: {e}")
			return False

		finally:
			time.sleep(0.1)
			self.last_successful_operation = time.time()

	def perform_task(self, task_description, max_steps=20):
		"""
		Execute a complete automation task using Claude's guidance.

		This is a legacy method maintained for compatibility. For new implementations,
		consider using ask_claude_default() which provides better error handling
		and user interaction.

		Args:
			task_description (str): Description of the task to accomplish
			max_steps (int): Maximum number of steps to attempt

		Returns:
			bool: True if task completed successfully, False otherwise
		"""
		if not self.connect_rdp():
			return False

		print(f"Starting task: {task_description}")
		actions_taken = []

		for step in range(max_steps):
			print(f"\n--- Step {step + 1} ---")

			try:
				screenshot = self.take_screenshot(silent=True)
				logger.debug("Screenshot captured successfully")

				action_type, params = self.ask_claude_for_action(
					screenshot,
					task_description,
					self.action_history[-5:]
				)

				if not action_type:
					print("Could not determine next action")
					continue

				if action_type == 'done':
					print("Task completed successfully!")
					break

				if self.execute_action(action_type, params):
					action_desc = f"{action_type}:{params}" if params else action_type
					actions_taken.append(action_desc)
				else:
					print("Failed to execute action")
					break

			except Exception as e:
				print(f"Error in step {step + 1}: {e}")
				break

		self.disconnect_rdp()
		return True

	def start_repl(self):
		"""
		Start the interactive Read-Eval-Print Loop (REPL) interface.

		Provides an interactive command-line interface for desktop automation
		with command history, tab completion, and rich formatted output.
		"""
		readline.set_startup_hook(None)
		readline.parse_and_bind("tab: complete")
		readline.parse_and_bind("set editing-mode emacs")

		history_file = os.path.join(user_data_dir("gnome-attendant"), "history.txt")

		# Ensure the directory exists
		history_dir = os.path.dirname(history_file)
		os.makedirs(history_dir, exist_ok=True)

		try:
			readline.read_history_file(history_file)
		except FileNotFoundError:
			pass

		self.show_banner()

		while True:
			try:
				status = "🟢 Connected" if self.connected else "🔴 Disconnected"

				try:
					command = input(f"\n🤖 Attendant ({status}) ❯ ").strip()
				except KeyboardInterrupt:
					self.console.print("\n⏹️  Ctrl+C pressed. Use '/quit' to exit the program.", style="yellow")
					continue
				except EOFError:
					self.console.print("\n\n👋 Goodbye!", style="green")
					self.disconnect_rdp()
					break

				if not command:
					continue

				readline.add_history(command)

				if command.startswith('/'):
					result = self.handle_slash_command(command[1:])
					if result == "exit":
						break
				else:
					if self.auto_connect_if_needed():
						self.ask_claude_default(command)
					else:
						self.console.print("❌ Could not establish RDP connection", style="red")

			except KeyboardInterrupt:
				self.console.print("\n⏹️  Operation cancelled with Ctrl+C", style="yellow")
				continue
			except Exception as e:
				self.console.print(f"❌ Error: {e}", style="red")

		try:
			readline.write_history_file(history_file)
		except:
			pass

	def handle_slash_command(self, command):
		"""
		Process and execute slash commands from the REPL interface.

		Args:
			command (str): Command string (without the leading slash)

		Returns:
			str or None: "exit" if the program should terminate, None otherwise
		"""
		try:
			parts = shlex.split(command.strip())
			cmd = parts[0].lower()
			args = parts[1:]

			if cmd in ['exit', 'quit', 'q']:
				self.console.print("Goodbye! 👋", style="green")
				return "exit"
			elif cmd == 'help':
				self.show_help()
			elif cmd == 'connect':
				self.cmd_connect(args)
			elif cmd == 'disconnect':
				self.cmd_disconnect()
			elif cmd == 'screenshot' or cmd == 'ss':
				self.cmd_screenshot(args)
			elif cmd == 'click':
				self.cmd_click(args)
			elif cmd == 'double_click' or cmd == 'double-click':
				self.cmd_double_click(args)
			elif cmd == 'right_click' or cmd == 'right-click':
				self.cmd_right_click(args)
			elif cmd == 'type':
				self.cmd_type(args)
			elif cmd == 'key':
				self.cmd_key(args)
			elif cmd == 'wait':
				self.cmd_wait(args)

			elif cmd == 'history':
				self.cmd_show_history()
			elif cmd == 'status':
				self.cmd_show_status()
			elif cmd == 'verbose':
				self.cmd_set_log_level('DEBUG')
			elif cmd == 'quiet':
				self.cmd_set_log_level('WARNING')
			elif cmd == 'debug':
				self.cmd_toggle_debug()
			elif cmd == 'host':
				self.cmd_set_host(args)
			elif cmd == 'password':
				self.cmd_set_password(args)
			elif cmd == 'username' or cmd == 'user':
				self.cmd_set_username(args)
			elif cmd == 'reconnect':
				self.cmd_reconnect(args)
			elif cmd == 'max_steps':
				self.cmd_set_max_steps(args)
			elif cmd == 'api_key':
				self.cmd_api_key(args)
			elif cmd == 'import':
				self.cmd_import(args)
			elif cmd == 'scale_coordinates':
				self.cmd_scale_coordinates(args)
			else:
				self.console.print(f"❌ Unknown command: /{cmd}. Type '/help' for available commands.", style="red")

		except KeyboardInterrupt:
			self.console.print("\n⏹️  Command cancelled with Ctrl+C", style="yellow")
		except ValueError as e:
			self.console.print(f"❌ Command parsing error: {e}", style="red")
		except Exception as e:
			self.console.print(f"❌ Error executing command: {e}", style="red")

	def ask_claude_default(self, task):
		"""
		Execute a multi-step automation task with Claude AI guidance.

		This is the primary method for intelligent desktop automation, supporting
		up to 25 automated steps with real-time progress feedback and error recovery.

		Args:
			task (str): Natural language description of the task to accomplish
		"""
		max_steps = self.max_steps
		step = 0

		self.console.print(f"🎯 Starting task: [bold]{task}[/bold]")
		self.console.print("[dim]💡 Press Ctrl+C to cancel current task (you can then start a new task)[/dim]")

		try:
			while step < max_steps:
				step += 1

				try:
					with self.console.status(f"🤖 Analyzing screen (step {step})..."):
						if step > 1:
							time.sleep(0.2)
						self.last_screenshot = self.take_screenshot(silent=True)

						action_type, params = self.ask_claude_for_action(
							self.last_screenshot,
							task,
							self.action_history[-5:]
						)

					if not action_type:
						self.console.print("🤖 Claude couldn't determine an action", style="yellow")
						self.add_action_to_history("unknown", None, False, "Claude couldn't determine action")
						break

					if action_type == 'done':
						self.console.print("✅ Task completed successfully!", style="green")
						self.add_action_to_history("done", None, True)
						break

					action_line = f"🤖 Step {step}: [bold cyan]{action_type}[/bold cyan]"
					if params:
						action_line += f" [yellow]{params}[/yellow]"
					self.console.print(action_line, end="")

					try:
						if self.execute_action(action_type, params):
							self.add_action_to_history(action_type, params, True)
							try:
								if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
									if sys.platform == 'win32':
										os.system('')
									self.console.file.write('\033[1A\033[2K\r')
									self.console.file.flush()
								else:
									self.console.print()
							except:
								self.console.print()

							success_line = f"✅ Step {step}: [bold cyan]{action_type}[/bold cyan]"
							if params:
								success_line += f" [yellow]{params}[/yellow]"
							self.console.print(success_line, style="green")

							time.sleep(0.5)
						else:
							self.add_action_to_history(action_type, params, False, "Action execution returned False")
							self.console.print("❌ Action failed", style="red")
					except Exception as action_error:
						self.add_action_to_history(action_type, params, False, str(action_error))
						self.console.print(f"❌ Action failed: {action_error}", style="red")

				except Exception as e:
					self.console.print(f"❌ Error in step {step}: {e}", style="red")
					self.add_action_to_history("error", None, False, str(e))
					break
			else:
				self.console.print(f"⚠️  Reached maximum steps ({max_steps}). Task may need more actions.", style="yellow")

		except KeyboardInterrupt:
			self.console.print(f"\n⏹️  Task aborted by user (completed {step-1} steps)", style="yellow")
			self.console.print("💡 You can continue with another task or use '/quit' to exit", style="dim")
			self.add_action_to_history("abort", f"step_{step}", False, "Task aborted by user with Ctrl+C")

	def show_banner(self):
		"""Display the welcome banner with configuration hints and session info."""
		config_hint = ""
		if not os.path.exists(self.CONFIG_FILE):
			config_hint = "\n[yellow]💡 First time? Use '/host' and '/password' to configure connection[/yellow]"

		restored_hint = ""
		if len(self.action_history) > 0:
			restored_hint = f"\n[green]🔄 Session restored with {len(self.action_history)} actions in history[/green]"

		banner = Panel.fit(
			"[bold blue]🤖 GNOME Attendant[/bold blue]\n"
			"[dim]Interactive RDP automation with Claude AI[/dim]\n\n"
			"Just type what you want to do (auto-connects & asks Claude)\n"
			"Use /command for other actions • Type '/help' for commands\n"
			"[dim]💡 Pipe commands to stdin for batch mode[/dim]" + config_hint + restored_hint,
			title="Welcome",
			border_style="blue"
		)
		self.console.print(banner)

	def show_help(self):
		"""Display comprehensive help information."""
		help_text = f"""
[bold cyan]GNOME Attendant - AI-Powered Desktop Automation[/bold cyan]

[bold yellow]QUICK START:[/bold yellow]
  1. Set connection: [cyan]/host server.domain.com:3389[/cyan]
  2. Set credentials: [cyan]/username myuser[/cyan] and [cyan]/password[/cyan]
  3. Connect: [cyan]/connect[/cyan]
  4. Use natural language: [cyan]click on the start button[/cyan]

[bold yellow]CONNECTION COMMANDS:[/bold yellow]
  [cyan]/host <hostname[:port]>[/cyan]     Set RDP server (port defaults to 3389)
  [cyan]/username <username>[/cyan]        Set authentication username
  [cyan]/password [password][/cyan]        Set password (prompts if omitted)
  [cyan]/domain <domain>[/cyan]            Set authentication domain
  [cyan]/api_key [key][/cyan]              Set Anthropic API key (prompts if omitted)
  [cyan]/connect[/cyan]                    Connect to RDP server
  [cyan]/disconnect[/cyan]                 Disconnect from RDP server
  [cyan]/reconnect[/cyan]                  Reconnect to RDP server
  [cyan]/status[/cyan]                     Show connection and system status

[bold yellow]AUTOMATION COMMANDS:[/bold yellow]
  [cyan]/screenshot[/cyan]                 Take a screenshot
  [cyan]/click <x> <y>[/cyan]              Click at coordinates
  [cyan]/type <text>[/cyan]                Type text
  [cyan]/key <keyname>[/cyan]              Press a key (e.g., Return, Tab, Escape)
  [cyan]/wait <seconds>[/cyan]             Wait for specified time

[bold yellow]AI COMMANDS:[/bold yellow]
  [cyan]<natural language>[/cyan]          Use AI to interpret and execute commands
  Example: "click on the start button", "open notepad", "type hello world"

[bold yellow]UTILITY COMMANDS:[/bold yellow]
  [cyan]/config[/cyan]                     Show current configuration
  [cyan]/history[/cyan]                    Show command history
  [cyan]/import <file>[/cyan]              Import and execute commands from file
  [cyan]/scale_coordinates factor <s>|reset[/cyan]  Down/up-scale AI screenshots and inverse-scale click coords
  [cyan]/clear[/cyan]                      Clear the screen
  [cyan]/debug[/cyan]                      Toggle debug logging
  [cyan]/help[/cyan]                       Show this help
  [cyan]/quit[/cyan]                       Exit GNOME Attendant

[bold yellow]USAGE MODES:[/bold yellow]
  [cyan]Interactive mode:[/cyan]           ./gnome-attendant
  [cyan]Batch mode:[/cyan]                 echo "commands" | ./gnome-attendant
  [cyan]Single command:[/cyan]             echo "/screenshot" | ./gnome-attendant

[bold yellow]TIPS:[/bold yellow]
  • Use natural language for complex tasks: "find the file menu and click save"
  • Screenshots are automatically taken before AI analysis
  • Configuration is saved automatically between sessions
  • Use /debug to see detailed operation logs
  • RDP connection is maintained automatically with reconnection
  • Import files can contain both slash commands and natural language
  • Lines starting with # are treated as comments and ignored
  • Use .attendant extension for automation scripts (e.g., hello.attendant)

[bold yellow]REQUIREMENTS:[/bold yellow]
   • Anthropic API key (use /api_key or ANTHROPIC_API_KEY env var)
   • FreeRDP3 libraries (libfreerdp3-dev)
   • Target machine running RDP server

[bold yellow]CONFIGURATION:[/bold yellow]
   • Config stored in: ~/.config/gnome-attendant/config.json
   • API keys stored securely in GNOME Keyring
   • Automatically created on first use
   • Follows XDG Base Directory specification

[dim]For more information, see the documentation or use specific command help.[/dim]
		"""

		self.console.print(Panel(help_text, title="Help", border_style="cyan"))

	def cmd_connect(self, args):
		"""Manually establish RDP server connection."""
		if self.connected:
			self.console.print("⚠️  Already connected. Disconnect first.", style="yellow")
			return

		try:
			with self.console.status("🔌 Connecting to RDP server..."):
				if self.connect_rdp():
					self.connected = True
					self.console.print("✅ Connected to RDP server!", style="green")
				else:
					self.console.print("❌ Failed to connect to RDP server", style="red")
		except KeyboardInterrupt:
			self.console.print("\n⏹️  Connection cancelled", style="yellow")

	def cmd_disconnect(self):
		"""Disconnect from RDP server."""
		if not self.connected:
			self.console.print("⚠️  Not currently connected to RDP server", style="yellow")
			return

		self.disconnect_rdp()
		self.console.print("✅ Disconnected from RDP server", style="green")

	def cmd_reconnect(self, args):
		"""Reconnect to the RDP server."""
		self.console.print("🔄 Reconnecting to RDP server...", style="blue")

		# Disconnect if currently connected
		if self.connected:
			self.disconnect_rdp()

		# Reconnect
		if self.connect_rdp():
			self.console.print("✅ Successfully reconnected to RDP server", style="green")
		else:
			self.console.print("❌ Failed to reconnect to RDP server", style="red")

	def cmd_screenshot(self, args):
		"""Capture and optionally save a screenshot."""
		if not self.auto_connect_if_needed():
			return

		try:
			with self.console.status("📸 Taking screenshot..."):
				screenshot = self.take_screenshot()
				self.last_screenshot = screenshot

			filename = args[0] if args else None
			if filename:
				screenshot.save(filename)
				self.console.print(f"📸 Screenshot saved as {filename}", style="green")
			else:
				self.console.print(f"📸 Screenshot taken ({screenshot.size[0]}x{screenshot.size[1]})", style="green")

		except KeyboardInterrupt:
			self.console.print("\n⏹️  Screenshot aborted", style="yellow")
		except Exception as e:
			self.console.print(f"❌ Screenshot failed: {e}", style="red")

	def cmd_click(self, args):
		"""Execute a mouse click action at specified coordinates."""
		if not self.auto_connect_if_needed():
			return

		if len(args) != 2:
			self.console.print("❌ Usage: click <x> <y>", style="red")
			return

		try:
			x, y = int(args[0]), int(args[1])
			if self.execute_action('click', (x, y)):
				self.add_action_to_history('click', (x, y), True)
				self.console.print(f"✅ Clicked at ({x}, {y})", style="green")
			else:
				self.add_action_to_history('click', (x, y), False, "Click execution failed")
				self.console.print("❌ Click failed", style="red")
		except ValueError:
			self.console.print("❌ Coordinates must be numbers", style="red")

	def cmd_double_click(self, args):
		"""Execute a mouse double-click action at specified coordinates."""
		if not self.auto_connect_if_needed():
			return

		if len(args) != 2:
			self.console.print("❌ Usage: double_click <x> <y>", style="red")
			return

		try:
			x, y = int(args[0]), int(args[1])
			if self.execute_action('double_click', (x, y)):
				self.add_action_to_history('double_click', (x, y), True)
				self.console.print(f"✅ Double-clicked at ({x}, {y})", style="green")
			else:
				self.add_action_to_history('double_click', (x, y), False, "Double-click execution failed")
				self.console.print("❌ Double-click failed", style="red")
		except ValueError:
			self.console.print("❌ Coordinates must be numbers", style="red")

	def cmd_right_click(self, args):
		"""Execute a mouse right-click action at specified coordinates."""
		if not self.auto_connect_if_needed():
			return

		if len(args) != 2:
			self.console.print("❌ Usage: right_click <x> <y>", style="red")
			return

		try:
			x, y = int(args[0]), int(args[1])
			if self.execute_action('right_click', (x, y)):
				self.add_action_to_history('right_click', (x, y), True)
				self.console.print(f"✅ Right-clicked at ({x}, {y})", style="green")
			else:
				self.add_action_to_history('right_click', (x, y), False, "Right-click execution failed")
				self.console.print("❌ Right-click failed", style="red")
		except ValueError:
			self.console.print("❌ Coordinates must be numbers", style="red")

	def cmd_type(self, args):
		"""Execute a text typing action."""
		if not self.auto_connect_if_needed():
			return

		if not args:
			self.console.print("❌ Usage: type <text>", style="red")
			return

		text = ' '.join(args).strip("'\"")
		if self.execute_action('type', text):
			self.add_action_to_history('type', text, True)
			self.console.print(f"✅ Typed: {text}", style="green")
		else:
			self.add_action_to_history('type', text, False, "Type execution failed")
			self.console.print("❌ Type failed", style="red")

	def cmd_key(self, args):
		"""Execute a keyboard key press action."""
		if not self.auto_connect_if_needed():
			return

		if not args:
			self.console.print("❌ Usage: key <keyname>", style="red")
			return

		key = args[0]
		if self.execute_action('key', key):
			self.add_action_to_history('key', key, True)
			self.console.print(f"✅ Pressed key: {key}", style="green")
		else:
			self.add_action_to_history('key', key, False, "Key press execution failed")
			self.console.print("❌ Key press failed", style="red")

	def cmd_wait(self, args):
		"""Execute a timed wait/pause action."""
		if not args:
			self.console.print("❌ Usage: wait <seconds>", style="red")
			return

		try:
			seconds = float(args[0])
			if self.execute_action('wait', seconds):
				self.add_action_to_history('wait', seconds, True)
				self.console.print(f"✅ Waited {seconds} seconds", style="green")
			else:
				self.add_action_to_history('wait', seconds, False, "Wait execution failed")
				self.console.print("❌ Wait failed", style="red")
		except ValueError:
			self.console.print("❌ Wait time must be a number", style="red")


	def cmd_show_history(self):
		"""Display recent action history with success/failure status."""
		if not self.detailed_action_history:
			self.console.print("📝 No actions in history", style="yellow")
			return

		history_table = Table(title="Action History", show_header=True, header_style="bold blue")
		history_table.add_column("#", width=4, style="cyan")
		history_table.add_column("Action", style="white")
		history_table.add_column("Status", width=12, style="white")
		history_table.add_column("Error", style="red")

		recent_actions = self.detailed_action_history[-10:]
		for i, action in enumerate(recent_actions, 1):
			status = "✅ SUCCESS" if action['success'] else "❌ FAILED"
			error_msg = action.get('error', '') if not action['success'] else ''
			history_table.add_row(str(i), action['action'], status, error_msg)

		self.console.print(history_table)

	def cmd_show_status(self):
		"""Display comprehensive system and connection status information."""
		if self.detailed_action_history:
			successful = sum(1 for action in self.detailed_action_history if action['success'])
			total = len(self.detailed_action_history)
			success_rate = (successful / total) * 100
			success_info = f"Success rate: {success_rate:.1f}% ({successful}/{total})"
		else:
			success_info = "No actions performed yet"

		current_level = logging.getLogger().getEffectiveLevel()
		level_names = {
			logging.DEBUG: "🔊 DEBUG",
			logging.INFO: "📝 INFO",
			logging.WARNING: "🔇 WARNING",
			logging.ERROR: "❌ ERROR"
		}
		log_level_display = level_names.get(current_level, f"❓ {current_level}")

		config_source = "📁 Saved config" if os.path.exists(self.CONFIG_FILE) else "⚙️  Default values"
		password_status = "🔐 Set" if self.rdp_password else "🔓 None"

		# Determine current framebuffer for display
		current_fb_display = "unknown"
		try:
			if self.freerdp_client and hasattr(self.freerdp_client, 'get_framebuffer_size'):
				current_framebuffer_width, current_framebuffer_height = self.freerdp_client.get_framebuffer_size()
				if current_framebuffer_width and current_framebuffer_height:
					current_fb_display = f"{current_framebuffer_width}x{current_framebuffer_height}"
		except Exception:
			pass

		status_panel = Panel(
			f"RDP Connection: {'🟢 Connected' if self.connected else '🔴 Disconnected'}\n"
			f"Host: {self.rdp_host}:{self.rdp_port} ({config_source})\n"
			f"Password: {password_status}\n"
			f"Config file: {self.CONFIG_FILE}\n"
			f"Log level: {log_level_display}\n"
			f"Current framebuffer: {current_fb_display}\n"
			f"Actions in history: {len(self.action_history)}\n"
			f"Detailed tracking: {len(self.detailed_action_history)} entries\n"
			f"Max steps per task: {self.max_steps}\n"
			f"Scale: {('max_width=' + str(self.screenshot_max_width)) if self.scale_strategy=='max_width' else ('factor=' + f'{self.screenshot_scale_factor:.3f}')} (last_effective={self.last_effective_scale_factor:.3f})\n"
			f"{success_info}\n"
			f"Last screenshot: {'📸 Available' if self.last_screenshot else '❌ None'}",
			title="Status",
			border_style="blue"
		)
		self.console.print(status_panel)

	def cmd_set_log_level(self, level):
		"""Configure the logging verbosity level."""
		level_map = {
			'DEBUG': logging.DEBUG,
			'INFO': logging.INFO,
			'WARNING': logging.WARNING,
			'ERROR': logging.ERROR
		}

		if level in level_map:
			logging.getLogger().setLevel(level_map[level])
			if level == 'DEBUG':
				self.console.print("🔊 Debug logging enabled - all messages will be shown", style="green")
			elif level == 'WARNING':
				self.console.print("🔇 Quiet mode enabled - only warnings and errors will be shown", style="green")
			else:
				self.console.print(f"📝 Logging level set to {level}", style="green")
		else:
			self.console.print("❌ Invalid log level. Use DEBUG, INFO, WARNING, or ERROR", style="red")

	def cmd_toggle_debug(self):
		"""Toggle debug logging between DEBUG and WARNING levels."""
		current_level = logging.getLogger().getEffectiveLevel()

		if current_level == logging.DEBUG:
			# Switch to WARNING (quiet mode)
			logging.getLogger().setLevel(logging.WARNING)
			self.console.print("🔇 Debug logging disabled - switched to quiet mode", style="yellow")
		else:
			# Switch to DEBUG
			logging.getLogger().setLevel(logging.DEBUG)
			self.console.print("🔊 Debug logging enabled - all messages will be shown", style="green")

	def cmd_set_host(self, args):
		"""Configure RDP server host and port settings."""
		if args:
			host_port = ' '.join(args)
			if ':' in host_port:
				try:
					host, port_str = host_port.rsplit(':', 1)
					port = int(port_str)
					self.rdp_host = host
					self.rdp_port = port
				except ValueError:
					self.console.print("❌ Invalid port number in host:port format", style="red")
					return
			else:
				self.rdp_host = host_port
		else:
			current_display = f"{self.rdp_host}:{self.rdp_port}"
			new_host_port = Prompt.ask(
				f"Enter RDP host:port",
				default=current_display,
				console=self.console
			)

			if ':' in new_host_port:
				try:
					host, port_str = new_host_port.rsplit(':', 1)
					port = int(port_str)
					self.rdp_host = host
					self.rdp_port = port
				except ValueError:
					self.console.print("❌ Invalid port number", style="red")
					return
			else:
				self.rdp_host = new_host_port

		if self.save_config():
			self.console.print(f"✅ RDP host set to: {self.rdp_host}:{self.rdp_port}", style="green")

			if self.connected:
				self.console.print("🔌 Disconnecting due to host change...", style="yellow")
				self.disconnect_rdp()
		else:
			self.console.print("⚠️  Host updated but failed to save configuration", style="yellow")

	def cmd_set_password(self, args):
		"""Configure RDP server password (secure input when no args provided)."""
		if args:
			self.rdp_password = ' '.join(args)
			# Only show warning in interactive mode (when stdin is a tty)
			if sys.stdin.isatty():
				self.console.print("⚠️  Warning: Password provided as command argument (visible in history)", style="yellow")
		else:
			self.console.print("🔐 Enter RDP password securely (typing will be hidden)")
			new_password = getpass.getpass("Password (empty for no password): ")
			self.rdp_password = new_password if new_password else None

		if self.save_config():
			if self.rdp_password:
				self.console.print("✅ RDP password set", style="green")
			else:
				self.console.print("✅ RDP password cleared", style="green")

			if self.connected:
				self.console.print("🔌 Disconnecting due to password change...", style="yellow")
				self.disconnect_rdp()

	def cmd_set_username(self, args):
		"""Configure RDP server username."""
		if args:
			self.rdp_username = ' '.join(args)
		else:
			new_username = input("Username (empty for no username): ")
			self.rdp_username = new_username if new_username else None

		if self.save_config():
			if self.rdp_username:
				self.console.print(f"✅ RDP username set to: {self.rdp_username}", style="green")
			else:
				self.console.print("✅ RDP username cleared", style="green")

			if self.connected:
				self.console.print("🔌 Disconnecting due to username change...", style="yellow")
				self.disconnect_rdp()
		else:
			self.console.print("⚠️  Password updated but failed to save configuration", style="yellow")

	def cmd_set_max_steps(self, args):
		"""Configure the maximum number of automation steps per task."""
		if args:
			try:
				new_max_steps = int(args[0])
				if new_max_steps < 1:
					self.console.print("❌ Maximum steps must be at least 1", style="red")
					return
				elif new_max_steps > 100:
					self.console.print("❌ Maximum steps cannot exceed 100", style="red")
					return

				self.max_steps = new_max_steps
			except ValueError:
				self.console.print("❌ Maximum steps must be a number", style="red")
				return
		else:
			try:
				new_max_steps = int(Prompt.ask(
					f"Enter maximum steps per task",
					default=str(self.max_steps),
					console=self.console
				))
				if new_max_steps < 1:
					self.console.print("❌ Maximum steps must be at least 1", style="red")
					return
				elif new_max_steps > 100:
					self.console.print("❌ Maximum steps cannot exceed 100", style="red")
					return

				self.max_steps = new_max_steps
			except ValueError:
				self.console.print("❌ Maximum steps must be a number", style="red")
				return
			except KeyboardInterrupt:
				self.console.print("\n⏹️  Max steps configuration cancelled", style="yellow")
				return

		if self.save_config():
			self.console.print(f"✅ Maximum steps set to: {self.max_steps}", style="green")
		else:
			self.console.print("⚠️  Maximum steps updated but failed to save configuration", style="yellow")

	def cmd_api_key(self, args):
		"""Set the Anthropic API key (stored securely in keyring)."""
		if args:
			api_key = args[0]
		else:
			try:
				api_key = getpass.getpass("Enter Anthropic API key: ")
			except KeyboardInterrupt:
				self.console.print("\n⏹️  API key setup cancelled", style="yellow")
				return

		if not api_key or not api_key.strip():
			self.console.print("❌ API key cannot be empty", style="red")
			return

		# Validate API key format (should start with sk-)
		api_key = api_key.strip()
		if not api_key.startswith('sk-'):
			self.console.print("⚠️  Warning: API key doesn't start with 'sk-'. This might be incorrect.", style="yellow")

		if self._set_api_key(api_key):
			# Test the API key by reinitializing the client
			try:
				self.anthropic_client = anthropic.Anthropic(api_key=api_key)
				self.console.print("✅ API key stored securely in keyring and validated", style="green")
			except Exception as e:
				self.console.print(f"⚠️  API key stored but validation failed: {e}", style="yellow")
		else:
			self.console.print("❌ Failed to store API key in keyring", style="red")

	def cmd_import(self, args):
		"""Import and execute commands from a file."""
		if not args:
			self.console.print("❌ Usage: /import <filename>", style="red")
			return

		filename = args[0]

		# Handle relative paths and expand user home directory
		if not os.path.isabs(filename):
			filename = os.path.abspath(filename)
		filename = os.path.expanduser(filename)

		if not os.path.exists(filename):
			self.console.print(f"❌ File not found: {filename}", style="red")
			return

		if not os.access(filename, os.R_OK):
			self.console.print(f"❌ Cannot read file: {filename}", style="red")
			return

		try:
			with open(filename, 'r', encoding='utf-8') as f:
				lines = f.readlines()

			# Filter out empty lines and comments
			commands = []
			for i, line in enumerate(lines, 1):
				line = line.strip()
				if line and not line.startswith('#'):
					commands.append((i, line))

			if not commands:
				self.console.print("⚠️  No executable commands found in file", style="yellow")
				return

			self.console.print(f"📂 Importing {len(commands)} command(s) from: {filename}", style="blue")

			for line_num, command in commands:
				try:
					# Sanitize password commands for display
					display_command = command
					if command.lower().startswith('/password '):
						display_command = '/password [HIDDEN]'
					self.console.print(f"[dim]L{line_num}:[/dim] {display_command}", style="cyan")

					# Process the command like the main REPL loop
					if command.startswith('/'):
						result = self.handle_slash_command(command[1:])
						if result == "exit":
							self.console.print("🛑 Import stopped due to exit command", style="yellow")
							break
					else:
						if self.auto_connect_if_needed():
							self.ask_claude_default(command)
						else:
							self.console.print("❌ Could not establish RDP connection", style="red")
							break

					# Small delay between commands to avoid overwhelming
					time.sleep(0.1)

				except KeyboardInterrupt:
					self.console.print(f"\n⏹️  Import cancelled at line {line_num}", style="yellow")
					break
				except Exception as e:
					self.console.print(f"❌ Error processing line {line_num}: {e}", style="red")
					# Ask user if they want to continue on error
					try:
						choice = Prompt.ask(
							"Continue importing remaining commands?",
							choices=["y", "n"],
							default="y",
							console=self.console
						)
						if choice.lower() != 'y':
							break
					except KeyboardInterrupt:
						self.console.print("\n⏹️  Import cancelled", style="yellow")
						break

			self.console.print("✅ Import completed", style="green")

		except Exception as e:
			self.console.print(f"❌ Error reading file: {e}", style="red")

	def run_batch_mode(self):
		"""
		Execute commands in batch mode by reading from stdin.

		Enables scriptable automation by processing a series of commands
		from standard input, useful for automated deployment and testing.

		Returns:
			bool: True if batch processing completed successfully, False otherwise
		"""
		self.console.print("🤖 Running in batch mode - reading commands from stdin", style="blue")

		try:
			commands: list[str] = []
			for line in sys.stdin:
				line = line.strip()
				if line:
					commands.append(line)

			if not commands:
				self.console.print("No commands provided", style="yellow")
				return True

			self.console.print(f"📝 Processing {len(commands)} command(s)", style="green")

			# Separate configuration commands from action commands
			config_command_types = {
				'host', 'username', 'user', 'password', 'domain', 'api_key', 'max_steps',
				'help', 'status', 'quit', 'exit', 'q', 'verbose', 'quiet', 'history',
				'scale_coordinates',
				'import'
			}

			config_commands: list[str] = []
			action_commands: list[str] = []
			natural_commands: list[str] = []

			for cmd in commands:
				if cmd.startswith('/'):
					cmd_type = cmd[1:].split()[0].lower()
					if cmd_type in config_command_types:
						config_commands.append(cmd)
					else:
						action_commands.append(cmd)
				else:
					natural_commands.append(cmd)

			# Process configuration commands first
			for command in config_commands:
				# Sanitize password commands for display
				display_command = command
				if command.lower().startswith('/password '):
					display_command = '/password [HIDDEN]'
				self.console.print(display_command, style="cyan")

				try:
					result = self.handle_slash_command(command[1:])
					if result == "exit":
						self.console.print("Exit command encountered, stopping batch processing", style="yellow")
						return True
				except Exception as e:
					self.console.print(f"❌ Error processing command '{command}': {e}", style="red")
					continue

			# Only attempt auto-connect if we have action commands or natural language commands
			needs_connection = bool(action_commands or natural_commands)
			if needs_connection:
				if not self.auto_connect_if_needed():
					self.console.print("❌ Failed to connect to RDP server", style="red")
					return False

				# Process action commands (slash commands that need connection)
				for command in action_commands:
					self.console.print(command, style="cyan")

					try:
						result = self.handle_slash_command(command[1:])
						if result == "exit":
							self.console.print("Exit command encountered, stopping batch processing", style="yellow")
							return True
					except Exception as e:
						self.console.print(f"❌ Error processing command '{command}': {e}", style="red")
						continue

				# Process natural language commands
				for command in natural_commands:
					self.console.print(command, style="cyan")

					try:
						self.ask_claude_default(command)
					except Exception as e:
						self.console.print(f"❌ Error processing command '{command}': {e}", style="red")
						continue

			self.console.print("\n✅ Batch processing completed", style="green")
			return True

		except Exception as e:
			self.console.print(f"❌ Error in batch mode: {e}", style="red")
			return False

	def cmd_scale_coordinates(self, args):
		"""Configure uniform scaling for AI screenshots and inverse click mapping.

		Usage:
		  /scale_coordinates factor <s>  # e.g., 0.5 to downscale, 2.0 to upscale
		  /scale_coordinates reset       # disable scaling (factor = 1)
		"""
		try:
			if not args:
				current = (
					f"max_width={self.screenshot_max_width}" if self.scale_strategy == 'max_width' else f"factor={self.screenshot_scale_factor:.3f}"
				)
				self.console.print(f"Current scaling: {current}\nUsage: /scale_coordinates factor <s> | max_width <px> | reset", style="blue")
				return

			mode = args[0].lower()
			if mode in ['reset', 'off', 'none']:
				self.screenshot_scale_factor = 1.0
				self.screenshot_max_width = None
				self.scale_strategy = 'factor'
				if self.save_config():
					self.console.print("✅ Coordinate scaling disabled", style="green")
				else:
					self.console.print("⚠️  Disabled scaling but failed to save configuration", style="yellow")
				return

			if mode in ['factor', 'scale']:
				if len(args) < 2:
					self.console.print("❌ Usage: /scale_coordinates factor <s>", style="red")
					return
				scale_factor_value = float(args[1])
				if scale_factor_value <= 0:
					self.console.print("❌ Scale factor must be > 0", style="red")
					return
				# Clamp to reasonable bounds
				if scale_factor_value < 0.05 or scale_factor_value > 8.0:
					self.console.print("⚠️  Large scale factor; clamping to [0.05, 8.0]", style="yellow")
					scale_factor_value = max(0.05, min(8.0, scale_factor_value))
				self.screenshot_scale_factor = scale_factor_value
				self.screenshot_max_width = None
				self.scale_strategy = 'factor'
				if self.save_config():
					self.console.print(f"✅ Set scale factor to {self.screenshot_scale_factor:.3f}", style="green")
				else:
					self.console.print("⚠️  Scale updated but failed to save configuration", style="yellow")
				return

			if mode in ['max_width', 'maxwidth', 'width']:
				if len(args) < 2:
					self.console.print("❌ Usage: /scale_coordinates max_width <px>", style="red")
					return
				max_width_pixels = int(args[1])
				if max_width_pixels <= 0:
					self.console.print("❌ Max width must be a positive integer", style="red")
					return
				# Clamp to reasonable bounds
				if max_width_pixels < 64 or max_width_pixels > 8192:
					self.console.print("⚠️  Max width clamped to [64, 8192]", style="yellow")
					max_width_pixels = max(64, min(8192, max_width_pixels))
				self.screenshot_max_width = max_width_pixels
				self.scale_strategy = 'max_width'
				if self.save_config():
					self.console.print(f"✅ Set max width to {self.screenshot_max_width}", style="green")
				else:
					self.console.print("⚠️  Max width updated but failed to save configuration", style="yellow")
				return

			self.console.print("❌ Unknown mode. Use 'factor', 'max_width', or 'reset'", style="red")
		except ValueError:
			self.console.print("❌ Invalid number format", style="red")
		except KeyboardInterrupt:
			self.console.print("\n⏹️  Scaling configuration cancelled", style="yellow")

	def _detect_local_screen_resolution(self) -> tuple[Optional[int], Optional[int]]:
		"""Detect the local primary screen resolution to request from the server.

		Attempts, in order:
		- xrandr (X11): parse the primary display line
		- xdpyinfo (X11): parse dimensions line
		Returns (width, height) or (None, None) if unknown.
		"""
		# Try xrandr first
		try:
			process = subprocess.run(
				["xrandr", "--current"],
				stdout=subprocess.PIPE,
				stderr=subprocess.DEVNULL,
				text=True,
				check=False,
			)
			output = process.stdout or ""
			for line in output.splitlines():
				line_stripped = line.strip()
				# Match primary line: e.g., "eDP-1 connected primary 3456x2160+0+0 ..."
				if " connected primary " in line_stripped:
					match = re.search(r"\b(\d+)x(\d+)\+\d+\+\d+", line_stripped)
					if match:
						return int(match.group(1)), int(match.group(2))
				# Fallback: first connected mode with resolution
				if " connected " in line_stripped:
					match = re.search(r"\b(\d+)x(\d+)\+\d+\+\d+", line_stripped)
					if match:
						return int(match.group(1)), int(match.group(2))
		except Exception:
			pass

		# Try xdpyinfo as a fallback
		try:
			process = subprocess.run(
				["xdpyinfo"],
				stdout=subprocess.PIPE,
				stderr=subprocess.DEVNULL,
				text=True,
				check=False,
			)
			for line in (process.stdout or "").splitlines():
				if "dimensions:" in line:
					match = re.search(r"dimensions:\s*(\d+)x(\d+)", line)
					if match:
						return int(match.group(1)), int(match.group(2))
		except Exception:
			pass

		return None, None

def main():
	"""Main entry point for GNOME Attendant."""
	# Logging is already configured at module level
	# Allow override via environment variable
	if os.getenv('GNOME_ATTENDANT_DEBUG'):
		logging.getLogger().setLevel(logging.DEBUG)

	# FreeRDP import is already handled at module level

	attendant = Attendant()

	# Check if running in batch mode (stdin has data)
	if not sys.stdin.isatty():
		attendant.run_batch_mode()
	else:
		attendant.start_repl()

if __name__ == "__main__":
	main()

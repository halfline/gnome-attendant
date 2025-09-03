#!/usr/bin/env python3
"""
Complete FreeRDP Python bindings using ctypes.

This implementation provides:
- Connection with SPNEGO/hybrid authentication (compatible with gnome-remote-desktop)
- Mouse movement and clicking
- Keyboard input
- Screen capture capabilities
- Proper settings configuration using FreeRDP's name-based API

Based on analysis of FreeRDP 3.16.0 headers and gnome-connections implementation.

Key settings choices:
- Software GDI: More reliable than hardware GDI across different systems
- Progressive GFX: Better performance with lower bandwidth
- H.264 disabled: More compatible, avoids codec issues
- Bitmap cache disabled: More reliable screen updates
"""

# Standard library imports
import ctypes
import logging
import threading
import time
from ctypes import (
    POINTER, Structure, CFUNCTYPE, c_void_p, c_char_p, c_int, c_uint32,
    c_int32, c_bool, c_size_t, byref, cast, string_at, c_uint8, c_uint16
)
from enum import Enum
from typing import Any, Callable, Optional, Tuple

# Local imports
from libfreerdp3_so_3.constants import (
    KBD_FLAGS_DOWN,
    KBD_FLAGS_RELEASE,
    KBD_FLAGS_EXTENDED,
    PTR_FLAGS_MOVE,
    PTR_FLAGS_DOWN,
    PTR_FLAGS_BUTTON1,
    PTR_FLAGS_BUTTON2,
    PTR_FLAGS_BUTTON3,
    RDP_CLIENT_INTERFACE_VERSION,
    DRDYNVC_CHANNEL_NAME,
    RDPGFX_DVC_CHANNEL_NAME,
)
from libfreerdp3_so_3.symbols import (
    freerdp_check_event_handles,
    freerdp_check_fds,
    freerdp_connect,
    freerdp_disconnect,
    freerdp_get_event_handles,
    freerdp_get_last_error,
    freerdp_get_last_error_string,
    freerdp_input_send_keyboard_event,
    freerdp_input_send_mouse_event,
    freerdp_input_send_keyboard_event_ex,
    freerdp_input_send_unicode_keyboard_event,
    freerdp_keyboard_get_rdp_scancode_from_x11_keycode,
    freerdp_settings_free,
    freerdp_settings_new,
    freerdp_settings_set_value_for_name,
    freerdp_shall_disconnect,
    gdi_free,
    gdi_get_pixel_format,
    gdi_graphics_pipeline_init,
    gdi_init,
    PubSub_OnEvent,
    freerdp_context_free,
)
from libfreerdp3_so_3.types import (
    rdp_context,
    freerdp,
    rdpInput,
    rdpSettings,
    rdp_update,
    rdpGdi,
)
from libfreerdp_client3_so_3.symbols import (
    freerdp_client_context_free,
    freerdp_client_context_new,
    freerdp_client_get_instance,
)
from libfreerdp_client3_so_3.types import RDP_CLIENT_ENTRY_POINTS_V1
from libwinpr3_so_3.symbols import (
    WaitForMultipleObjects,
    PubSub_Subscribe,
    GetVirtualKeyCodeFromKeycode,
    GetVirtualScanCodeFromVirtualKeyCode,
    GetVirtualKeyCodeFromName,
)
from libwinpr3_so_3.constants import KBDEXT
from libwinpr3_so_3.types import (
    DWORD,
    BOOL,
    UINT32,
    UINT16,
    UINT8,
    INT32,
    HANDLE,
)

# Constants from winpr/input.h
WINPR_KBD_TYPE_IBM_ENHANCED = 0x00000004
WINPR_KEYCODE_TYPE_XKB = 0x00000003

logger = logging.getLogger(__name__)

class WEventArgs(Structure):
    _fields_ = [("Size", DWORD), ("Sender", c_char_p)]

class ResizeWindowEventArgs(Structure):
    _fields_ = [("e", WEventArgs), ("width", UINT32), ("height", UINT32)]

class ChannelConnectedEventArgs(Structure):
    _fields_ = [("e", WEventArgs), ("name", c_char_p), ("pInterface", c_void_p)]

class DynamicChannelConnectedEventArgs(Structure):
    _fields_ = [("e", WEventArgs), ("name", c_char_p), ("pInterface", c_void_p)]

class Rect16(Structure):
    _fields_ = [
        ("left", UINT16),
        ("top", UINT16),
        ("right", UINT16),
        ("bottom", UINT16),
    ]

class RDP_CLIENT_CONTEXT(Structure):
    _pack_ = 8
    _fields_ = [
        ("context", rdp_context),
    ]

# Callback type definitions
RdpGlobalInitCallback = CFUNCTYPE(BOOL)
RdpGlobalUninitCallback = CFUNCTYPE(None)
RdpClientNewCallback = CFUNCTYPE(BOOL, POINTER(freerdp), POINTER(rdp_context))
RdpClientFreeCallback = CFUNCTYPE(None, POINTER(freerdp), POINTER(rdp_context))
RdpClientStartCallback = CFUNCTYPE(c_int, POINTER(rdp_context))
RdpClientStopCallback = CFUNCTYPE(c_int, POINTER(rdp_context))

BeginPaintCallback = CFUNCTYPE(BOOL, POINTER(rdp_context))
EndPaintCallback = CFUNCTYPE(BOOL, POINTER(rdp_context))
DesktopResizeCallback = CFUNCTYPE(BOOL, POINTER(rdp_context))
RefreshRectCallback = CFUNCTYPE(BOOL, POINTER(rdp_context), UINT8, POINTER(Rect16))
BitmapUpdateCallback = CFUNCTYPE(None, POINTER(rdp_context), c_void_p)

AuthenticateCallback = CFUNCTYPE(BOOL, POINTER(freerdp), POINTER(c_char_p), POINTER(c_char_p), POINTER(c_char_p), c_int)
VerifyCertificateCallback = CFUNCTYPE(DWORD, POINTER(freerdp), c_char_p, UINT16, c_char_p, c_char_p, c_char_p, c_char_p, DWORD)
LogonErrorCallback = CFUNCTYPE(None, POINTER(freerdp), UINT32, c_char_p)
PreConnectCallback = CFUNCTYPE(BOOL, POINTER(freerdp))
PostConnectCallback = CFUNCTYPE(BOOL, POINTER(freerdp))
PostDisconnectCallback = CFUNCTYPE(None, POINTER(freerdp))

# Keep PubSub subscription handlers alive for the duration of the program
_pubsub_subscription_handlers: list[Any] = []


# Create callback function pointers at module level since they need to stay alive

# Text entry uses Unicode keyboard events.



# Utility helpers

def _initialize_function_prototypes():
    """Set up function prototypes for imported symbols."""
    WaitForMultipleObjects.restype = DWORD
    WaitForMultipleObjects.argtypes = [DWORD, POINTER(HANDLE), BOOL, DWORD]

    PubSub_Subscribe.restype = c_int
    PubSub_Subscribe.argtypes = [c_void_p, c_char_p, CFUNCTYPE(c_int, c_void_p, POINTER(WEventArgs))]

    PubSub_OnEvent.restype = c_int
    PubSub_OnEvent.argtypes = [c_void_p, c_char_p, c_void_p, POINTER(WEventArgs)]

    freerdp_get_event_handles.restype = c_uint32
    freerdp_get_event_handles.argtypes = [POINTER(rdp_context), POINTER(HANDLE), c_uint32]

    freerdp_client_context_new.restype = POINTER(RDP_CLIENT_CONTEXT)
    freerdp_client_context_new.argtypes = [POINTER(RDP_CLIENT_ENTRY_POINTS_V1)]

    freerdp_client_context_free.restype = None
    freerdp_client_context_free.argtypes = [POINTER(RDP_CLIENT_CONTEXT)]

    freerdp_client_get_instance.restype = POINTER(freerdp)
    freerdp_client_get_instance.argtypes = [POINTER(RDP_CLIENT_CONTEXT)]

    freerdp_context_free.restype = None
    freerdp_context_free.argtypes = [POINTER(freerdp)]

    freerdp_connect.restype = BOOL
    freerdp_connect.argtypes = [POINTER(freerdp)]

    freerdp_disconnect.restype = BOOL
    freerdp_disconnect.argtypes = [POINTER(freerdp)]

    freerdp_shall_disconnect.restype = BOOL
    freerdp_shall_disconnect.argtypes = [POINTER(freerdp)]

    freerdp_check_fds.restype = c_int
    freerdp_check_fds.argtypes = [POINTER(freerdp)]

    freerdp_check_event_handles.restype = BOOL
    freerdp_check_event_handles.argtypes = [POINTER(rdp_context)]

    freerdp_settings_new.restype = POINTER(rdpSettings)
    freerdp_settings_new.argtypes = [DWORD]

    freerdp_settings_free.restype = None
    freerdp_settings_free.argtypes = [POINTER(rdpSettings)]

    freerdp_settings_set_value_for_name.restype = BOOL
    freerdp_settings_set_value_for_name.argtypes = [POINTER(rdpSettings), c_char_p, c_char_p]

    freerdp_get_last_error.restype = c_uint32
    freerdp_get_last_error.argtypes = [POINTER(rdp_context)]

    freerdp_get_last_error_string.restype = c_char_p
    freerdp_get_last_error_string.argtypes = [UINT32]

    gdi_init.restype = BOOL
    gdi_init.argtypes = [POINTER(freerdp), UINT32]

    gdi_free.restype = None
    gdi_free.argtypes = [POINTER(freerdp)]

    gdi_get_pixel_format.restype = UINT32
    gdi_get_pixel_format.argtypes = [UINT32]

    freerdp_input_send_mouse_event.restype = BOOL
    freerdp_input_send_mouse_event.argtypes = [POINTER(rdpInput), UINT16, UINT16, UINT16]

    freerdp_input_send_keyboard_event.restype = BOOL
    freerdp_input_send_keyboard_event.argtypes = [POINTER(rdpInput), UINT16, UINT8]
    freerdp_input_send_keyboard_event_ex.restype = BOOL
    freerdp_input_send_keyboard_event_ex.argtypes = [POINTER(rdpInput), BOOL, BOOL, UINT32]
    freerdp_input_send_unicode_keyboard_event.restype = BOOL
    freerdp_input_send_unicode_keyboard_event.argtypes = [POINTER(rdpInput), UINT16, UINT16]

    gdi_graphics_pipeline_init.restype = BOOL
    gdi_graphics_pipeline_init.argtypes = [c_void_p, c_void_p]

    GetVirtualKeyCodeFromKeycode.restype = UINT32
    GetVirtualKeyCodeFromKeycode.argtypes = [UINT32, UINT32]
    GetVirtualScanCodeFromVirtualKeyCode.restype = UINT32
    GetVirtualScanCodeFromVirtualKeyCode.argtypes = [UINT32, UINT32]
    GetVirtualKeyCodeFromName.restype = UINT32
    GetVirtualKeyCodeFromName.argtypes = [c_char_p]
    freerdp_keyboard_get_rdp_scancode_from_x11_keycode.restype = UINT32
    freerdp_keyboard_get_rdp_scancode_from_x11_keycode.argtypes = [UINT32]


_initialize_function_prototypes()


def PubSub_SubscribeEvent(pubsub: c_void_p, event_name: bytes, handler: "Callable[[c_void_p, POINTER(WEventArgs)], None]") -> tuple[int, Any]:
    callback_type = CFUNCTYPE(c_int, c_void_p, POINTER(WEventArgs))
    callback = callback_type(handler)
    result = PubSub_Subscribe(pubsub, event_name, callback)
    _pubsub_subscription_handlers.append(callback)
    return (int(result) if result is not None else 0, callback)




def client_authenticate_ex(instance, username, password, domain, reason):
    """Handle AuthenticateEx requests - provide actual credentials"""
    if username:
        username.contents = c_char_p(b"testuser")
    if password:
        password.contents = c_char_p(b"testpass")
    if domain:
        domain.contents = c_char_p(b"")
    return True


def client_verify_certificate_ex(instance, host, port, common_name, subject, issuer, fingerprint, flags):
    """Handle certificate verification - accept for testing"""
    return CertificateVerification.ACCEPT_TEMPORARILY.value


def client_logon_error_info(instance, data, message):
    """Handle logon error information"""
    logger.debug(f"LogonErrorInfo callback: data={data}, message={message}")

# RDPGFX logging callbacks

def client_pre_connect(instance):
    """Pre-connection callback"""
    context_pointer = cast(instance.contents.context, POINTER(rdp_context))
    context = context_pointer.contents
    pubsub_pointer = cast(context.pubSub, c_void_p) if context.pubSub else None

    if pubsub_pointer:
        def _on_channel_connected(context_void, event_args_pointer):
            connected_args = cast(event_args_pointer, POINTER(ChannelConnectedEventArgs)).contents
            name_bytes = connected_args.name if connected_args.name else None
            channel_name = name_bytes.decode(errors='ignore') if name_bytes else ''
            if channel_name == '':
                return 0
            if channel_name == DRDYNVC_CHANNEL_NAME:
                interface = connected_args.pInterface
                logger.info(f"DVC manager interface {interface:x}" if interface else "None")
            elif channel_name == RDPGFX_DVC_CHANNEL_NAME:
                interface = connected_args.pInterface
                logger.info(f"Graphics channel {channel_name} connected with interface {interface:x}" if interface else "None")
                init_ok = bool(gdi_graphics_pipeline_init(context.gdi, interface))
                if not init_ok:
                    logger.info("gdi_graphics_pipeline_init (on rdpgfx channel connected) failed")
            else:
                interface = connected_args.pInterface
                logger.info(f"Other channel connected: {channel_name} (interface: {interface:x})" if interface else "None")
            return 0

        result, _ = PubSub_SubscribeEvent(pubsub_pointer, b"ChannelConnected", _on_channel_connected)
        logger.info(f"PreConnect: Subscribed ChannelConnected result={result}")
    return True


def client_post_connect(instance):
    """Post-connection callback"""
    logger.info("PostConnect callback invoked - connection established")
    return True


def client_post_disconnect(instance):
    """Post-disconnect callback - logging only, no cleanup here"""
    logger.info("PostDisconnect callback - connection closed")


def _client_global_init():
    """Global initialization - must return BOOL"""
    return True


def _client_global_uninit():
    """Global cleanup - can be empty"""
    pass


def _client_new(instance, context):
    """Create new client context - attach callbacks"""
    try:
        inst = instance.contents
        inst.PreConnect = cast(pre_connect_callback, c_void_p)
        inst.PostConnect = cast(post_connect_callback, c_void_p)
        inst.PostDisconnect = cast(post_disconnect_callback, c_void_p)
        inst.AuthenticateEx = cast(authenticate_ex_callback, c_void_p)
        inst.VerifyCertificateEx = cast(verify_cert_callback, c_void_p)
        inst.LogonErrorInfo = cast(logon_error_callback, c_void_p)

        return True
    except Exception as error:
        logger.error(f"Client new error: {error}")
        return False


def _client_free(instance, context):
    """Free client context - correct signature with both args"""
    return None


def _client_start(context):
    """Start client"""
    return 0


def _client_stop(context):
    """Stop client"""
    return 0

# Assign callback function pointers AFTER definitions
_GLOBAL_INIT_CB = RdpGlobalInitCallback(_client_global_init)
_GLOBAL_UNINIT_CB = RdpGlobalUninitCallback(_client_global_uninit)
_CLIENT_NEW_CB = RdpClientNewCallback(_client_new)
_CLIENT_FREE_CB = RdpClientFreeCallback(_client_free)
_CLIENT_START_CB = RdpClientStartCallback(_client_start)
_CLIENT_STOP_CB = RdpClientStopCallback(_client_stop)

# Helper function to create pre-configured settings and entry points
def create_configured_entry_points(hostname="", port=3389, username="", password="",
                               domain="", width: Optional[int] = None, height: Optional[int] = None):
    """Create RDP_CLIENT_ENTRY_POINTS_V1 with pre-configured settings.
    
    The settings object is owned by the returned entry points structure.
    The caller must ensure the settings are freed when no longer needed.
    
    Args:
        hostname: Target RDP server hostname
        port: Target RDP server port
        username: RDP username
        password: RDP password
        domain: RDP domain
        width: Desktop width in pixels
        height: Desktop height in pixels
        
    Returns:
        tuple[RDP_CLIENT_ENTRY_POINTS_V1, POINTER(rdpSettings)]: Entry points and settings
        
    Raises:
        FreeRDPException: If settings creation fails
    """
    # Create settings object FIRST and configure it fully
    settings = freerdp_settings_new(0)
    if not settings:
        raise FreeRDPException("Failed to create settings")

    # Enable graphics pipeline and software GDI
    success = True
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_SupportGraphicsPipeline", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_SoftwareGdi", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_GfxProgressive", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_GfxH264", b"false"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_GfxThinClient", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_RemoteFxCodec", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_BitmapCacheEnabled", b"false"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_DrawAllowSkipAlpha", b"false"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_ColorDepth", b"32"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_NetworkAutoDetect", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_SupportDynamicTimeZone", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_SupportHeartbeatPdu", b"true"))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_DynamicResolutionUpdate", b"true"))
    # Enable Display Control (monitor layout) channel so the server can send/display layout updates
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_SupportDisplayControl", b"true"))
    # Enable RefreshRect callbacks so we can copy only damaged regions
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_RefreshRect", b"true"))


    # Connection parameters
    if hostname:
        success &= bool(freerdp_settings_set_value_for_name(
            settings, b"FreeRDP_ServerHostname", hostname.encode('utf-8')))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_ServerPort", str(port).encode('utf-8')))
    if username:
        success &= bool(freerdp_settings_set_value_for_name(
            settings, b"FreeRDP_Username", username.encode('utf-8')))
    if password:
        success &= bool(freerdp_settings_set_value_for_name(
            settings, b"FreeRDP_Password", password.encode('utf-8')))
    if domain:
        success &= bool(freerdp_settings_set_value_for_name(
            settings, b"FreeRDP_Domain", domain.encode('utf-8')))
    if width is not None and height is not None:
        success &= bool(freerdp_settings_set_value_for_name(
            settings, b"FreeRDP_DesktopWidth", str(width).encode('utf-8')))
        success &= bool(freerdp_settings_set_value_for_name(
            settings, b"FreeRDP_DesktopHeight", str(height).encode('utf-8')))
    success &= bool(freerdp_settings_set_value_for_name(
        settings, b"FreeRDP_IgnoreCertificate", b"true"))

    if success:
        logger.info("GFX settings: pipeline=true, software_gdi=true, progressive=true, thin_client=true, color_depth=32, bitmap_cache=false")
    else:
        logger.warning("Some settings may have failed")

    # Create entry points structure with configured settings
    entry = RDP_CLIENT_ENTRY_POINTS_V1()
    entry.Version = RDP_CLIENT_INTERFACE_VERSION
    entry.Size = ctypes.sizeof(RDP_CLIENT_ENTRY_POINTS_V1)
    entry.settings = cast(settings, c_void_p)  # Cast settings pointer to c_void_p
    # Cast function pointers to c_void_p as required by entry points structure
    entry.GlobalInit = cast(_GLOBAL_INIT_CB, c_void_p)
    entry.GlobalUninit = cast(_GLOBAL_UNINIT_CB, c_void_p)
    entry.ContextSize = ctypes.sizeof(RDP_CLIENT_CONTEXT)  # Must match our client context
    entry.ClientNew = cast(_CLIENT_NEW_CB, c_void_p)
    entry.ClientFree = cast(_CLIENT_FREE_CB, c_void_p)
    entry.ClientStart = cast(_CLIENT_START_CB, c_void_p)
    entry.ClientStop = cast(_CLIENT_STOP_CB, c_void_p)

    return entry, settings  # Return both for reference

# Create callback function pointers
authenticate_ex_callback = AuthenticateCallback(client_authenticate_ex)
verify_cert_callback = VerifyCertificateCallback(client_verify_certificate_ex)
logon_error_callback = LogonErrorCallback(client_logon_error_info)
pre_connect_callback = PreConnectCallback(client_pre_connect)
post_connect_callback = PostConnectCallback(client_post_connect)
post_disconnect_callback = PostDisconnectCallback(client_post_disconnect)


# Event types for RDP events
class RDPEventType(Enum):
    """Event types for RDP connection events."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SCREEN_UPDATE = "screen_update"
    ERROR = "error"

class CertificateVerification(Enum):
    """Certificate verification status codes.
    
    These values match FreeRDP's internal verification constants:
    - DENY: Certificate is not trusted/denied
    - ACCEPT_PERMANENTLY: Certificate is permanently trusted
    - ACCEPT_TEMPORARILY: Certificate is temporarily trusted for this session
    """
    DENY = 0
    ACCEPT_PERMANENTLY = 1
    ACCEPT_TEMPORARILY = 2

class FreeRDPException(Exception):
    """Exception raised for FreeRDP errors."""
    pass

# Alias for backward compatibility
FreeRDPError = FreeRDPException


class ScreenCapture:
    """Screen capture data container."""
    def __init__(self, width: int, height: int, data: bytes, format: str = 'BGRX'):
        self.width = width
        self.height = height
        self.data = data
        self.format = format

    def save_as_ppm(self, filename: str):
        """Save as PPM image file."""
        with open(filename, 'wb') as f:
            f.write(f"P6\n{self.width} {self.height}\n255\n".encode())
            # Data is already in RGB format from get_screenshot
            if self.format == "RGB":
                f.write(self.data)
            else:
                # Fallback for other formats - convert BGRA/BGRX to RGB
                rgb_data = bytearray()
                for i in range(0, len(self.data), 4):
                    if i + 3 < len(self.data):
                        b, g, r = self.data[i], self.data[i+1], self.data[i+2]
                        rgb_data.extend([r, g, b])
                f.write(rgb_data)

    def save_raw(self, filename: str):
        """Save raw BGRX data."""
        with open(filename, 'wb') as f:
            f.write(self.data)


class FreeRDPClient:
    def __init__(self):
        self._context = None
        self._instance = None
        self._settings = None
        self._connected = False
        self._gdi_ready = False
        self._rdp_context = None  # Replaces _ctx_p

        self._event_thread = None
        self._event_thread_stop = threading.Event()

        self._shadow_buffer = None
        self._shadow_width = 0
        self._shadow_height = 0
        self._shadow_stride = 0
        self._shadow_format = "BGRX"
        self._shadow_bottom_up = False
        self._shadow_frame_id = 0
        self._shadow_lock = threading.RLock()

        # Replace individual callback fields with dicts
        self._original_callbacks = {
            'begin_paint': None,
            'end_paint': None,
            'desktop_resize': None,
            'refresh_rect': None,
            'bitmap_update': None,
        }
        
        self._keepalive_callbacks = {
            'begin_paint': None,
            'end_paint': None,
            'desktop_resize': None,
            'refresh_rect': None,
            'bitmap_update': None,
        }

        # Remove individual callback fields
        self._callback_begin_paint = None
        self._callback_end_paint = None
        self._callback_desktop_resize = None

        self._begin_paint_count = 0
        self._end_paint_count = 0
        self._desktop_resize_count = 0

        self._auto_dump_ppm = False
        self._auto_dump_path = 'dump.ppm'

        self._diagnostic_log = True
        self._last_handle_count = -1
        self._last_wait_result = -9999

    @staticmethod
    def _coerce_uint32(value: Any) -> int:
        try:
            if isinstance(value, int):
                return value & 0xFFFFFFFF
            if hasattr(value, 'value'):
                return int(value.value) & 0xFFFFFFFF
            if isinstance(value, (bytes, bytearray)):
                return int.from_bytes(value, byteorder='little', signed=False) & 0xFFFFFFFF
            return int(value) & 0xFFFFFFFF
        except Exception:
            return 0

    def __del__(self):
        self.disconnect()
        if self._context:
            try:
                freerdp_client_context_free(self._context)
            except Exception:
                pass

    def _alloc_shadow_from_gdi(self, gdi) -> None:
        width = gdi.width
        height = gdi.height
        stride = gdi.stride
        if width <= 0 or height <= 0:
            return
        with self._shadow_lock:
            self._shadow_width = width
            self._shadow_height = height
            self._shadow_stride = abs(stride)
            self._shadow_bottom_up = (stride < 0)
            size = self._shadow_stride * self._shadow_height
            if self._shadow_buffer is None or len(self._shadow_buffer) != size:
                self._shadow_buffer = bytearray(size)

    def _on_begin_paint(self, context_pointer) -> bool:
        self._begin_paint_count += 1
        if self._begin_paint_count <= 5 or self._begin_paint_count % 100 == 0:
            logger.debug(f"BeginPaint callback #{self._begin_paint_count}")
        if self._original_callbacks['begin_paint']:
            return bool(self._original_callbacks['begin_paint'](context_pointer))
        return True

    def _on_end_paint(self, context_pointer) -> bool:
        self._end_paint_count += 1
        if self._end_paint_count <= 5 or self._end_paint_count % 100 == 0:
            logger.debug(f"EndPaint callback #{self._end_paint_count}")

        context = context_pointer.contents
        if not context.gdi:
            return True
        gdi = cast(context.gdi, POINTER(rdpGdi)).contents

        self._alloc_shadow_from_gdi(gdi)

        if self._shadow_buffer is None:
            return True

        # Fallback: copy the entire primary buffer if partial-refresh path is inactive
        if gdi.primary_buffer:
            size = self._shadow_stride * self._shadow_height
            raw = string_at(gdi.primary_buffer, size)
            with self._shadow_lock:
                memoryview(self._shadow_buffer)[:size] = raw
                self._shadow_frame_id = gdi.frameId
        else:
            with self._shadow_lock:
                self._shadow_frame_id = gdi.frameId

        if self._auto_dump_ppm and self._shadow_buffer is not None:
            logger.info(f"Dumping shadow to {self._auto_dump_path}")
            self._dump_shadow_to_ppm(self._auto_dump_path)

        if self._original_callbacks['end_paint']:
            return bool(self._original_callbacks['end_paint'](context_pointer))
        return True

    def _on_desktop_resize(self, context_pointer) -> bool:
        self._desktop_resize_count += 1
        if self._end_paint_count <= 5 or self._end_paint_count % 100 == 0:
            logger.debug(f"DesktopResize callback #{self._desktop_resize_count}")
        context = context_pointer.contents
        if not context.gdi:
            return True
        gdi = cast(context.gdi, POINTER(rdpGdi)).contents
        old_size = f"{self._shadow_width}x{self._shadow_height}" if self._shadow_buffer else "none"
        self._alloc_shadow_from_gdi(gdi)
        new_size = f"{self._shadow_width}x{self._shadow_height}"
        logger.debug(f"DesktopResize #{self._desktop_resize_count}: {old_size} → {new_size}")
        try:
            settings = cast(context.settings, POINTER(rdpSettings)).contents if context.settings else None
            if settings:
                logger.info(f"DesktopResize applied. GDI now {self._shadow_width}x{self._shadow_height}")
        except Exception:
            pass
        if self._original_callbacks['desktop_resize']:
            return bool(self._original_callbacks['desktop_resize'](context_pointer))
        return True

    def _on_refresh_rect(self, context_pointer, count, rects_pointer) -> bool:
        logger.info(f"RefreshRect callback #{self._refresh_rect_count}")
        context = context_pointer.contents
        if not context.gdi:
            return True
        gdi = cast(context.gdi, POINTER(rdpGdi)).contents
        self._alloc_shadow_from_gdi(gdi)
        if self._shadow_buffer is None or not gdi.primary_buffer or count <= 0 or not rects_pointer:
            return True
        base_addr = int(gdi.primary_buffer)
        bytes_per_pixel = 4
        stride = self._shadow_stride
        height = self._shadow_height
        bottom_up = self._shadow_bottom_up
        with self._shadow_lock:
            view = memoryview(self._shadow_buffer)
            for i in range(int(count)):
                rect = rects_pointer[i]
                left = int(rect.left)
                top = int(rect.top)
                right = int(rect.right)
                bottom = int(rect.bottom)
                width_px = max(0, right - left)
                height_px = max(0, bottom - top)
                if width_px == 0 or height_px == 0:
                    continue
                left = max(0, min(left, self._shadow_width))
                top = max(0, min(top, self._shadow_height))
                right = max(left, min(right, self._shadow_width))
                bottom = max(top, min(bottom, self._shadow_height))
                width_px = right - left
                height_px = bottom - top
                row_bytes = width_px * bytes_per_pixel
                for row in range(height_px):
                    src_y = top + row
                    phys_y = (height - 1 - src_y) if bottom_up else src_y
                    src_offset = phys_y * stride + left * bytes_per_pixel
                    src_ptr = base_addr + src_offset
                    row_data = string_at(src_ptr, row_bytes)
                    view[src_offset:src_offset + row_bytes] = row_data
        self._shadow_frame_id = gdi.frameId
        if self._original_callbacks['refresh_rect']:
            return bool(self._original_callbacks['refresh_rect'](context_pointer, count, rects_pointer))
        return True

    def _on_bitmap_update(self, context_pointer, bitmap_update_ptr) -> None:
        try:
            self._refresh_rect_count = getattr(self, '_refresh_rect_count', 0)
            self._refresh_rect_count += 1
            logger.info(f"BitmapUpdate callback #{self._refresh_rect_count}")
        except Exception:
            pass

    def _install_update_hooks(self, context) -> None:
        """Install update hooks for GDI callbacks."""
        update_ptr = cast(context.update, POINTER(rdp_update))
        update = update_ptr.contents

        # Convert original function pointers to callable trampolines
        self._original_callbacks['begin_paint'] = BeginPaintCallback(update.BeginPaint) if update.BeginPaint else None
        self._original_callbacks['end_paint'] = EndPaintCallback(update.EndPaint) if update.EndPaint else None
        self._original_callbacks['desktop_resize'] = DesktopResizeCallback(update.DesktopResize) if update.DesktopResize else None
        self._original_callbacks['refresh_rect'] = RefreshRectCallback(update.RefreshRect) if update.RefreshRect else None
        self._original_callbacks['bitmap_update'] = BitmapUpdateCallback(update.BitmapUpdate) if update.BitmapUpdate else None

        # Create our callback functions
        self._keepalive_callbacks['begin_paint'] = BeginPaintCallback(self._on_begin_paint)
        self._keepalive_callbacks['end_paint'] = EndPaintCallback(self._on_end_paint)
        self._keepalive_callbacks['desktop_resize'] = DesktopResizeCallback(self._on_desktop_resize)
        self._keepalive_callbacks['refresh_rect'] = RefreshRectCallback(self._on_refresh_rect)
        self._keepalive_callbacks['bitmap_update'] = BitmapUpdateCallback(self._on_bitmap_update)

        # Install our callbacks
        update.BeginPaint = cast(self._keepalive_callbacks['begin_paint'], c_void_p)
        update.EndPaint = cast(self._keepalive_callbacks['end_paint'], c_void_p)
        update.DesktopResize = cast(self._keepalive_callbacks['desktop_resize'], c_void_p)
        update.RefreshRect = cast(self._keepalive_callbacks['refresh_rect'], c_void_p)
        update.BitmapUpdate = cast(self._keepalive_callbacks['bitmap_update'], c_void_p)

    def _start_event_pump_thread(self) -> None:
        if self._event_thread and self._event_thread.is_alive():
            return
        self._event_thread_stop.clear()
        self._event_thread = threading.Thread(target=self._event_pump_loop, name="FreeRDPEventPump", daemon=True)
        self._event_thread.start()

    def _stop_event_pump_thread(self) -> None:
        if not self._event_thread:
            return
        self._event_thread_stop.set()
        self._event_thread.join(timeout=1.0)
        self._event_thread = None

    def _event_pump_loop(self) -> None:
        """Event pump loop that handles RDP events and maintains connection state."""
        MAX_HANDLES = 64
        events_array = (HANDLE * MAX_HANDLES)()
        error_count = 0
        last_error_time = 0
        backoff = 0.02

        while not self._event_thread_stop.is_set():
            try:
                if not self._instance or not self._connected:
                    time.sleep(0.05)
                    continue

                context_pointer = self._rdp_context
                if context_pointer is None:
                    context_void_pointer = getattr(self._instance.contents, 'context', None)
                    if context_void_pointer:
                        context_pointer = cast(context_void_pointer, POINTER(rdp_context))
                        self._rdp_context = context_pointer

                if context_pointer is None:
                    time.sleep(0.05)
                    continue

                handle_count = int(freerdp_get_event_handles(context_pointer, events_array, c_uint32(MAX_HANDLES)))
                if handle_count < 0:
                    handle_count = 0

                if self._diagnostic_log and handle_count != self._last_handle_count:
                    logger.info(f"event loop: handle_count={handle_count}")
                    self._last_handle_count = handle_count

                if handle_count == 0:
                    time.sleep(0.05)
                    fds_result = freerdp_check_fds(self._instance)
                    if fds_result <= 0:
                        error_code = freerdp_get_last_error(context_pointer)
                        error_msg = freerdp_get_last_error_string(error_code)
                        now = time.time()
                        if now - last_error_time > 1:
                            logger.warning(f"Transport issue: {error_msg.decode('utf-8') if error_msg else 'unknown error'} (code {error_code})")
                            last_error_time = now
                        backoff = min(backoff * 1.5, 0.25)
                        time.sleep(backoff)
                    continue

                wait_result = WaitForMultipleObjects(DWORD(handle_count), events_array, BOOL(False), DWORD(100))
                if self._diagnostic_log and wait_result != self._last_wait_result:
                    self._last_wait_result = wait_result

                pumped = bool(freerdp_check_event_handles(context_pointer))
                if not pumped:
                    fds_result = freerdp_check_fds(self._instance)
                    if fds_result <= 0 and context_pointer:
                        error_code = freerdp_get_last_error(context_pointer)
                        error_msg = freerdp_get_last_error_string(error_code)
                        now = time.time()
                        if now - last_error_time > 1:
                            logger.warning(f"Transport issue: {error_msg.decode('utf-8') if error_msg else 'unknown error'} (code {error_code})")
                            last_error_time = now
                        backoff = min(backoff * 1.5, 0.25)
                        time.sleep(backoff)
                        continue
                    if fds_result == 0:
                        time.sleep(0.02)

                if bool(freerdp_shall_disconnect(self._instance)):
                    break

                error_count = 0
                backoff = 0.02

            except Exception as error:
                error_count += 1
                now = time.time()
                if now - last_error_time > 5:
                    logger.warning(f"Event pump loop error (count={error_count}): {error}")
                    last_error_time = now
                backoff = min(backoff * 1.5, 0.25)
                time.sleep(backoff)

        self._connected = False

    def get_live_frame(self):
        """Get the current screen frame data.
        Returns tuple (buffer, width, height, stride, format, bottom_up, frame_id) or None.
        """
        if self._shadow_buffer is None:
            return None
        with self._shadow_lock:
            return (
                self._shadow_buffer,
                self._shadow_width,
                self._shadow_height,
                self._shadow_stride,
                self._shadow_format,
                self._shadow_bottom_up,
                self._shadow_frame_id,
            )

    def get_framebuffer_size(self) -> tuple[int, int]:
        """Return current framebuffer width and height.
        Thread-safe snapshot of the latest known GDI dimensions.
        """
        with self._shadow_lock:
            return self._shadow_width, self._shadow_height

    def get_callback_stats(self):
        return {
            'begin_paint_calls': self._begin_paint_count,
            'end_paint_calls': self._end_paint_count,
            'desktop_resize_calls': self._desktop_resize_count,
            'shadow_frame_id': self._shadow_frame_id,
            'shadow_size': f"{self._shadow_width}x{self._shadow_height}" if self._shadow_buffer else "none"
        }

    def _pump_events(self, seconds: float):
        # Avoid manual pumps when the background event thread is active
        if self._event_thread and self._event_thread.is_alive():
            time.sleep(seconds)
            return 0
        time_end = time.time() + seconds
        events_received = 0
        while time.time() < time_end:
            result = freerdp_check_fds(self._instance)
            if result <= 0:
                time.sleep(0.01)
            else:
                events_received += 1
                time.sleep(0.005)
        return events_received

    def connect(self, hostname: str, port: int = 3389, username: str = "",
                password: str = "", domain: str = "",
                width: Optional[int] = None, height: Optional[int] = None,
                timeout: float = 30.0) -> bool:
        if self._connected:
            return True

        try:
            logger.info(f"Creating pre-configured client context for {hostname}:{port}...")
            entry, self._settings = create_configured_entry_points(
                hostname, port, username, password, domain, width, height)

            self._context = freerdp_client_context_new(byref(entry))
            if not self._context:
                if self._settings:
                    freerdp_settings_free(self._settings)
                    self._settings = None
                raise FreeRDPException("freerdp_client_context_new failed")

            self._instance = freerdp_client_get_instance(self._context)
            if not self._instance:
                freerdp_client_context_free(self._context)
                raise FreeRDPException("freerdp_client_get_instance failed")

            logger.info("Client context created with pre-configured settings")

            logger.info(f"About to call freerdp_connect to {hostname}:{port}...")
            logger.info(f"Instance: {self._instance}, Context: {self._context}")

            connect_result = freerdp_connect(self._instance)
            logger.info(f"freerdp_connect returned: {bool(connect_result)}")

            if not connect_result:
                context_ptr = cast(self._instance.contents.context, POINTER(rdp_context))
                if not context_ptr:
                    raise FreeRDPException("Connection failed: no context available")
                error_code = freerdp_get_last_error(context_ptr)
                error_msg = freerdp_get_last_error_string(error_code)
                raise FreeRDPException(f"Connection failed: {error_msg.decode('utf-8') if error_msg else 'unknown error'}")

            self._rdp_context = cast(self._instance.contents.context, POINTER(rdp_context))
            if not self._rdp_context:
                raise FreeRDPException("No context available after successful connect")

            self._connected = True
            logger.info("Connected successfully with GFX pipeline and software GDI")

            pixel_format = gdi_get_pixel_format(32)
            gdi_success = bool(gdi_init(self._instance, pixel_format))
            if not gdi_success:
                logger.error("gdi_init failed")
                return False

            client_context_ptr = cast(self._context, POINTER(RDP_CLIENT_CONTEXT))
            context = client_context_ptr.contents.context
            if not context.update:
                logger.error("rdpContext.update is NULL")
                return False

            self._install_update_hooks(context)

            gdi_ptr = context.gdi
            if not gdi_ptr:
                logger.error("GDI pointer is NULL after gdi_init")
                return False
            gdi = cast(gdi_ptr, POINTER(rdpGdi)).contents
            self._alloc_shadow_from_gdi(gdi)
            if gdi.primary_buffer and self._shadow_buffer is not None:
                size = self._shadow_stride * self._shadow_height
                raw0 = string_at(gdi.primary_buffer, size)
                with self._shadow_lock:
                    memoryview(self._shadow_buffer)[:size] = raw0
                    self._shadow_frame_id = gdi.frameId

            if not context.gdi:
                logger.error("GDI context is NULL after gdi_init")
                return False

            self._gdi_ready = True
            logger.info("GDI initialized successfully")

            self._start_event_pump_thread()

            return True

        except Exception as error:
            logger.error(f"Connection error: {error}")
            if self._context:
                try:
                    freerdp_client_context_free(self._context)
                    self._context = None
                except Exception:
                    pass
            return False

    def disconnect(self):
        if not self._connected or not self._instance:
            return

        self._stop_event_pump_thread()
        time.sleep(0.1)
        freerdp_disconnect(self._instance)
        self._connected = False

        if self._gdi_ready and self._instance:
            gdi_free(self._instance)
            self._gdi_ready = False

        self._settings = None
        logger.info("Disconnected")

    def is_connected(self) -> bool:
        if not self._connected:
            return False
        try:
            return not bool(freerdp_shall_disconnect(self._instance))
        except:
            return False

    def get_screenshot(self) -> Optional[ScreenCapture]:
        """Get a screenshot of the current display."""
        if not self._connected:
            return None

        frame_data = self.get_live_frame()
        if frame_data is None:
            return None

        view, width, height, stride, fmt, bottom_up, frame_id = frame_data
        if frame_id == 0:
            if self._event_thread and self._event_thread.is_alive():
                time.sleep(0.2)
                frame_data = self.get_live_frame()
            else:
                self._pump_events(0.2)
                frame_data = self.get_live_frame()
            if frame_data is None:
                return None
            view, width, height, stride, fmt, bottom_up, frame_id = frame_data

        if width <= 0 or height <= 0:
            return None

        min_stride = width * 3
        max_stride = width * 8
        if stride < min_stride or stride > max_stride:
            return None

        rgb = bytearray(width * height * 3)
        di = 0

        for y in range(height):
            si = (height - 1 - y) * stride if bottom_up else y * stride
            for x in range(width):
                b = view[si + 4*x + 0]
                g = view[si + 4*x + 1]
                r = view[si + 4*x + 2]
                rgb[di+0] = r
                rgb[di+1] = g
                rgb[di+2] = b
                di += 3

        return ScreenCapture(width, height, bytes(rgb), "RGB")

    def _dump_shadow_to_ppm(self, filename: str) -> None:
        with self._shadow_lock:
            if self._shadow_buffer is None or self._shadow_width <= 0 or self._shadow_height <= 0:
                return
            buffer = self._shadow_buffer
            width = self._shadow_width
            height = self._shadow_height
            stride = self._shadow_stride
            bottom_up = self._shadow_bottom_up

            header = f"P6\n{width} {height}\n255\n".encode()
            rgb_data = bytearray(width * height * 3)
            output_index = 0
            rows = range(height - 1, -1, -1) if bottom_up else range(0, height)
            for y in rows:
                row_offset = y * stride
                for x in range(0, width):
                    offset = row_offset + x * 4
                    blue = buffer[offset + 0]
                    green = buffer[offset + 1]
                    red = buffer[offset + 2]
                    rgb_data[output_index + 0] = red
                    rgb_data[output_index + 1] = green
                    rgb_data[output_index + 2] = blue
                    output_index += 3
            with open(filename, "wb") as output_file:
                output_file.write(header)
                output_file.write(rgb_data)

    def move_mouse(self, x: int, y: int) -> bool:
        if not self._connected or not self._context:
            return False

        try:
            client_context = cast(self._context, POINTER(RDP_CLIENT_CONTEXT)).contents
            context = client_context.context
            if not context.input:
                logger.error("Input context is NULL")
                return False

            input_obj = cast(context.input, POINTER(rdpInput))

            success = bool(freerdp_input_send_mouse_event(input_obj, PTR_FLAGS_MOVE, x, y))
            if success:
                logger.debug(f"Moved mouse to ({x}, {y})")
            else:
                logger.error(f"Failed to move mouse to ({x}, {y})")
            return success
        except Exception as error:
            logger.error(f"Mouse move error: {error}")
            return False

    def click_mouse(self, x: int, y: int, button: int = 1, double_click: bool = False) -> bool:
        if not self._connected or not self._context:
            return False

        try:
            client_context = cast(self._context, POINTER(RDP_CLIENT_CONTEXT)).contents
            context = client_context.context
            if not context.input:
                logger.error("Input context is NULL")
                return False

            input_obj = cast(context.input, POINTER(rdpInput))

            move_success = bool(freerdp_input_send_mouse_event(input_obj, PTR_FLAGS_MOVE, x, y))
            if not move_success:
                logger.error(f"Failed to move mouse to ({x}, {y}) before click")
                return False

            button_flags = {
                1: PTR_FLAGS_BUTTON1,
                2: PTR_FLAGS_BUTTON2,
                3: PTR_FLAGS_BUTTON3
            }

            if button not in button_flags:
                logger.error(f"Invalid button number: {button}")
                return False

            button_flag = button_flags[button]

            down_success = bool(freerdp_input_send_mouse_event(
                input_obj,
                PTR_FLAGS_DOWN | button_flag,
                x, y
            ))

            if not down_success:
                logger.error(f"Failed to send mouse down event")
                return False

            time.sleep(0.02)

            up_success = bool(freerdp_input_send_mouse_event(
                input_obj,
                button_flag,
                x, y
            ))

            if not up_success:
                logger.error(f"Failed to send mouse up event")
                return False

            if double_click:
                time.sleep(0.05)
                down_success2 = bool(freerdp_input_send_mouse_event(
                    input_obj,
                    PTR_FLAGS_DOWN | button_flag,
                    x, y
                ))
                if down_success2:
                    time.sleep(0.02)
                    up_success2 = bool(freerdp_input_send_mouse_event(
                        input_obj,
                        button_flag,
                        x, y
                    ))
                    if not up_success2:
                        logger.error(f"Failed to send second mouse up event for double-click")
                        return False
                else:
                    logger.error(f"Failed to send second mouse down event for double-click")
                    return False

            action = "Double-clicked" if double_click else "Clicked"
            logger.debug(f"{action} button {button} at ({x}, {y})")
            return True

        except Exception as error:
            logger.error(f"Mouse click error: {error}")
            return False

    def send_key(self, scancode: int, pressed: bool = True) -> bool:
        """Send a keyboard event using RDP scancode via extended API.
        Accepts Set 1 scancode (0..0xFF) or extended scancode with EXT_FLAG_MASK.
        """
        if not self._connected or not self._context:
            logger.error("Cannot send key - not connected or no context")
            return False
        try:
            client_context = cast(self._context, POINTER(RDP_CLIENT_CONTEXT)).contents
            context = client_context.context
            if not context.input:
                logger.error("Input context is NULL")
                return False
            input_pointer = cast(context.input, POINTER(rdpInput))
            rdp_scancode_value = UINT32(scancode & 0xFFFF)
            return bool(freerdp_input_send_keyboard_event_ex(input_pointer, BOOL(pressed), BOOL(False), rdp_scancode_value))
        except Exception as error:
            logger.error(f"Keyboard error: {error}")
            return False

    def send_key_name(self, key_name: str, pressed: bool = True, keyboard_type: int = WINPR_KBD_TYPE_IBM_ENHANCED) -> bool:
        """Send a key by virtual key name using WinPR -> RDP scancode mapping.
        Accepts names like 'VK_RIGHT', 'Right', 'A', '1', 'Enter', etc.
        """
        try:
            vk_code_hint, vk_name_hint = _normalize_virtual_key_name_for_winpr(key_name)
            # Try WinPR name first if we have one
            virtual_key_code = 0
            if vk_name_hint is not None:
                vk_raw = GetVirtualKeyCodeFromName(vk_name_hint.encode('ascii', errors='ignore'))
                virtual_key_code = self._coerce_uint32(vk_raw)
            # Fallback to direct VK code hint (ASCII A..Z/0..9 or mapped control/arrows)
            if virtual_key_code == 0 and vk_code_hint is not None:
                virtual_key_code = int(vk_code_hint)
            # Final attempt: try WinPR with provided name string directly
            if virtual_key_code == 0:
                vk_raw2 = GetVirtualKeyCodeFromName(key_name.encode('ascii', errors='ignore'))
                virtual_key_code = self._coerce_uint32(vk_raw2)
            if virtual_key_code == 0:
                logger.error(f"Unknown key name: {key_name}")
                return False
            return self.send_virtual_keycode(int(virtual_key_code), pressed, keyboard_type)
        except Exception as error:
            logger.error(f"send_key_name error: {error}")
            return False

    def send_virtual_scancode(self, virtual_scancode: int, pressed: bool, repeat: bool = False) -> bool:
        """Send a key using FreeRDP's extended scancode API (rdp_scancode).
        virtual_scancode: RDP scancode as defined by FreeRDP (DWORD).
        """
        if not self._connected or not self._context:
            logger.error("Cannot send key - not connected or no context")
            return False
        try:
            client_context = cast(self._context, POINTER(RDP_CLIENT_CONTEXT)).contents
            context = client_context.context
            if not context.input:
                logger.error("Input context is NULL")
                return False
            input_pointer = cast(context.input, POINTER(rdpInput))
            success = bool(
                freerdp_input_send_keyboard_event_ex(
                    input_pointer,
                    BOOL(1 if pressed else 0),
                    BOOL(1 if repeat else 0),
                    UINT32(virtual_scancode),
                )
            )
            return success
        except Exception as error:
            logger.error(f"send_virtual_scancode error: {error}")
            return False

    def send_virtual_keycode(self, virtual_keycode: int, pressed: bool, keyboard_type: int = WINPR_KBD_TYPE_IBM_ENHANCED) -> bool:
        """Convert a virtual key code to an RDP scancode and send via extended API."""
        try:
            # Use legacy API for extended keys (proved to work reliably)
            if virtual_keycode in _EXTENDED_VK_CODES:
                try:
                    client_context = cast(self._context, POINTER(RDP_CLIENT_CONTEXT)).contents
                    context = client_context.context
                    if not context.input:
                        logger.error("Input context is NULL")
                        return False
                    input_pointer = cast(context.input, POINTER(rdpInput))
                except Exception as get_input_error:
                    logger.error(f"Could not access input context: {get_input_error}")
                    return False

                set1_code = _EXTENDED_SET1_SCANCODE.get(virtual_keycode)
                if set1_code is None:
                    # Fallback: try WinPR mapping to get low 8-bit code
                    alt_raw = GetVirtualScanCodeFromVirtualKeyCode(UINT32(virtual_keycode), UINT32(keyboard_type))
                    alt = self._coerce_uint32(alt_raw) & 0xFF
                    set1_code = alt if alt != 0xFF and alt != 0 else 0x00
                if set1_code == 0x00:
                    logger.error(f"No Set-1 scancode for VK 0x{virtual_keycode:02X}")
                    return False

                flags = (KBD_FLAGS_DOWN if pressed else KBD_FLAGS_RELEASE) | KBD_FLAGS_EXTENDED
                ok = bool(freerdp_input_send_keyboard_event(input_pointer, UINT16(flags), UINT8(set1_code)))
                logger.info(f"(legacy) VK 0x{virtual_keycode:02X} -> set1 0x{set1_code:02X} EXT -> sent={ok}")
                return ok

            rdp_scan_raw = GetVirtualScanCodeFromVirtualKeyCode(UINT32(virtual_keycode), UINT32(keyboard_type))
            rdp_scan = self._coerce_uint32(rdp_scan_raw)
            if rdp_scan == 0xFF or rdp_scan == 0:
                # Fallback: resolve via FreeRDP's keyboard mapping from X11 keycode if available
                # X11 keycodes for arrows are typically 111 (Up), 113 (Left), 114 (Right), 116 (Down)
                x11_hint = {
                    0x25: 113,  # VK_LEFT -> XK_Left
                    0x26: 111,  # VK_UP -> XK_Up
                    0x27: 114,  # VK_RIGHT -> XK_Right
                    0x28: 116,  # VK_DOWN -> XK_Down
                }.get(virtual_keycode)
                if x11_hint is not None:
                    alt = self._coerce_uint32(freerdp_keyboard_get_rdp_scancode_from_x11_keycode(UINT32(x11_hint)))
                    if alt not in (0, 0xFF):
                        rdp_scan = alt
            if rdp_scan == 0 or rdp_scan == 0xFF:
                logger.error(f"Unknown virtual keycode: {virtual_keycode}")
                return False
            if (rdp_scan & 0xFF00) == 0 and virtual_keycode in _EXTENDED_VK_CODES:
                rdp_scan = (KBDEXT | (rdp_scan & 0xFF))
            logger.info(f"VK 0x{virtual_keycode:02X} -> RDP scan=0x{rdp_scan:04X}")
            return self.send_virtual_scancode(int(rdp_scan), pressed)
        except Exception as error:
            logger.error(f"send_virtual_keycode error: {error}")
            return False

    def send_text(self, text: str):
        """Send text input using Unicode keyboard events (UTF-16 code units).
        This avoids layout-specific scancode mapping and handles punctuation
        and casing without synthesizing modifier keys.
        """
        if not self._connected or not self._context:
            logger.error("Cannot send text - not connected or no context")
            return
        try:
            client_context = cast(self._context, POINTER(RDP_CLIENT_CONTEXT)).contents
            context = client_context.context
            if not context.input:
                logger.error("Input context is NULL")
                return
            input_pointer = cast(context.input, POINTER(rdpInput))
            for ch in text:
                unicode_code_point = ord(ch)
                if unicode_code_point > 0xFFFF:
                    logger.warning(f"Skipping non-BMP character U+{unicode_code_point:04X}")
                    continue
                down_ok = bool(freerdp_input_send_unicode_keyboard_event(input_pointer, KBD_FLAGS_DOWN, UINT16(unicode_code_point)))
                time.sleep(0.01)
                up_ok = bool(freerdp_input_send_unicode_keyboard_event(input_pointer, KBD_FLAGS_RELEASE, UINT16(unicode_code_point)))
                if not (down_ok and up_ok):
                    logger.warning(f"Failed to send unicode key U+{unicode_code_point:04X}")
                time.sleep(0.01)
        except Exception as error:
            logger.error(f"send_text error: {error}")


# Common virtual-key constants (Windows VK codes)
_VK_CODE_MAP: dict[str, int] = {
	"VK_RETURN": 0x0D,
	"VK_ENTER": 0x0D,
	"VK_TAB": 0x09,
	"VK_ESCAPE": 0x1B,
	"VK_ESC": 0x1B,
	"VK_SPACE": 0x20,
	"VK_BACK": 0x08,
	"VK_LEFT": 0x25,
	"VK_UP": 0x26,
	"VK_RIGHT": 0x27,
	"VK_DOWN": 0x28,
	"VK_INSERT": 0x2D,
	"VK_DELETE": 0x2E,
	"VK_HOME": 0x24,
	"VK_END": 0x23,
	"VK_PRIOR": 0x21,  # Page Up
	"VK_NEXT": 0x22,   # Page Down
}

_EXTENDED_VK_CODES = {
	0x25,  # LEFT
	0x26,  # UP
	0x27,  # RIGHT
	0x28,  # DOWN
	0x2D,  # INSERT
	0x2E,  # DELETE
	0x24,  # HOME
	0x23,  # END
	0x21,  # PRIOR (Page Up)
	0x22,  # NEXT (Page Down)
}

# Set-1 scancode map for extended keys (arrow cluster and navigation)
_EXTENDED_SET1_SCANCODE: dict[int, int] = {
	0x25: 0x4B,  # VK_LEFT
	0x26: 0x48,  # VK_UP
	0x27: 0x4D,  # VK_RIGHT
	0x28: 0x50,  # VK_DOWN
	0x2D: 0x52,  # VK_INSERT
	0x2E: 0x53,  # VK_DELETE
	0x24: 0x47,  # VK_HOME
	0x23: 0x4F,  # VK_END
	0x21: 0x49,  # VK_PRIOR (Page Up)
	0x22: 0x51,  # VK_NEXT (Page Down)
}

def _normalize_virtual_key_name_for_winpr(key_name: str) -> tuple[Optional[int], Optional[str]]:
	"""Normalize a key name to a VK code or a canonical WinPR VK name.
	Returns (vk_code_int, vk_name_str). One of them may be None.
	"""
	name = key_name.strip()
	if len(name) == 1:
		ch = name.upper()
		if 'A' <= ch <= 'Z' or '0' <= ch <= '9':
			return (ord(ch), None)
	alias_map = {
		"right": "VK_RIGHT",
		"left": "VK_LEFT",
		"up": "VK_UP",
		"down": "VK_DOWN",
		"return": "VK_RETURN",
		"enter": "VK_ENTER",
		"space": "VK_SPACE",
		"tab": "VK_TAB",
		"escape": "VK_ESCAPE",
		"esc": "VK_ESC",
		"backspace": "VK_BACK",
	}
	vk_name = alias_map.get(name.lower())
	if vk_name:
		vk_code = _VK_CODE_MAP.get(vk_name)
		return (vk_code, vk_name)
	vk_guess = f"VK_{name.upper()}"
	vk_code = _VK_CODE_MAP.get(vk_guess)
	if vk_code is not None:
		return (vk_code, vk_guess)
	return (None, vk_guess)


def example_usage():
    """Example usage of the FreeRDP client class."""
    logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more verbose output
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logger.debug = print
    logger.info = print
    
    # Create client instance
    client = FreeRDPClient()

    try:
        print("FreeRDP Python Bindings")
        print("=====================")
        print()
        print("FreeRDP Settings (GFX Pipeline + Software GDI):")
        print("  SupportGraphicsPipeline=true")
        print("  SoftwareGdi=true")
        print("  GfxProgressive=true")
        print("  GfxThinClient=true")
        print("  ColorDepth=32")
        print("  BitmapCacheEnabled=false")
        print()

        # Connect to gnome-remote-desktop
        success = client.connect(
            hostname="localhost",
            port=3390,
            username="testuser",
            password="testpass",
            width=1920,
            height=1080
        )
        time.sleep(1000000)
        if success:
            print("Connected to RDP server with GFX pipeline and software GDI")

            print("\n" + "="*50)
            print("KEYBOARD INPUT TESTS")
            print("Note: You may see key characters appear between debug messages -")
            print("this is normal and shows the keys are being processed!")
            print("="*50 + "\n")

            # Temporarily disable debug logging for cleaner output
            prev_level = logger.getEffectiveLevel()
            logger.setLevel(logging.WARNING)

            # Test keyboard input first
            print("TEST 1: Basic key press/release")
            print("-" * 30)
            # Test a simple 'A' keypress
            print("Sending 'A' key...")
            print("Expected output: 'a'")
            print("Actual output: ", end="", flush=True)
            key_down = client.send_key_name('A', True)
            time.sleep(0.1)
            key_up = client.send_key_name('A', False)
            print("\nKey 'A' down result: {}, up result: {}".format(key_down, key_up))
            print()

            # Test extended key (e.g. arrow key)
            print("TEST 2: Extended key (Arrow)")
            print("-" * 30)
            print("Sending RIGHT ARROW key (scancode 0x4D with extended flag)...")
            print("Expected output: cursor movement")
            print("Actual output: ", end="", flush=True)
            # For extended keys, we use the extended flag and different scancodes
            ext_down = client.send_key_name('Right', True)
            time.sleep(0.1)
            ext_up = client.send_key_name('Right', False)
            print("\nExtended key RIGHT ARROW down result: {}, up result: {}".format(ext_down, ext_up))
            print()

            # Test text input
            print("TEST 3: Text input")
            print("-" * 30)
            test_text = "Hello123!"
            print(f"Sending text: {test_text}")
            print("Expected output: 'Hello123!'")
            print("Actual output: ", end="", flush=True)
            for char in test_text:
                client.send_text(char)
            print("\nText sending complete")
            print()

            # Test 4: Repeated Return keys
            print("TEST 4: Repeated Return keys")
            print("-" * 30)
            print("Sending 20 Return keys... (you should see 20 newlines if focused in a shell)")
            for _ in range(20):
                client.send_key_name('Return', True)
                time.sleep(0.02)
                client.send_key_name('Return', False)
                time.sleep(0.02)
            print("Return key spam complete")
            print()

            # TEST 5: Repeated Up arrow keys
            print("TEST 5: Repeated UP ARROW keys")
            print("-" * 30)
            print("Sending 20 Up Arrow keys... (in a shell running cat -v you should see ^[[A)")
            for _ in range(20):
                client.send_key_name('Up', True)
                time.sleep(0.02)
                client.send_key_name('Up', False)
                time.sleep(0.02)
            print("Up Arrow key spam complete")
            print()

            # Restore debug logging
            logger.setLevel(prev_level)

            # Verify input context
            try:
                client_context = cast(client._context, POINTER(RDP_CLIENT_CONTEXT)).contents
                context = client_context.context
                if context.input:
                    input_obj = cast(context.input, POINTER(rdpInput))
                    print("Input context verification:")
                    print(f"Context pointer: {client._context}")
                    print(f"Input context available: {bool(context.input)}")
                    print(f"Input object pointer: {input_obj}")
                else:
                    print("\nWARNING: Input context is NULL!")
            except Exception as e:
                print(f"\nError checking input context: {e}")

            print("\n" + "="*50)
            print("KEYBOARD TESTS COMPLETE")
            print("="*50)

            print("\nDisconnecting...")
            client.disconnect()
            print("Disconnected")

        else:
            print("✗ Connection failed")
            print("Make sure gnome-remote-desktop is running on localhost:3390")

    except FreeRDPException as e:
        print(f"FreeRDP error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    print("FreeRDP Python Bindings")
    print("=====================")
    print()
    # Initialize prototypes now that the module is fully loaded
    _initialize_function_prototypes()
    example_usage()


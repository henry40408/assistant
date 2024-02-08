import logging

# This is a minimal configuration to get you started with the Text mode.
# If you want to connect Errbot to chat services, checkout
# the options in the more complete config-template.py from here:
# https://raw.githubusercontent.com/errbotio/errbot/master/errbot/config-template.py

BACKEND = "Text"  # Errbot will start in text mode (console only mode) and will answer commands from there.

BOT_DATA_DIR = r"/home/nixos/Develop/assistant/data"
BOT_EXTRA_PLUGIN_DIR = r"/home/nixos/Develop/assistant/plugins"
BOT_EXTRA_BACKEND_DIR = r"/home/nixos/Develop/assistant/backend-plugins"

BOT_LOG_FILE = r"/home/nixos/Develop/assistant/errbot.log"
BOT_LOG_LEVEL = logging.INFO

BOT_ADMINS = (
    "@CHANGE_ME",
)  # Don't leave this as "@CHANGE_ME" if you connect your errbot to a chat system!!

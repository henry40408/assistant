import logging
import os

# This is a minimal configuration to get you started with the Text mode.
# If you want to connect Errbot to chat services, checkout
# the options in the more complete config-template.py from here:
# https://raw.githubusercontent.com/errbotio/errbot/master/errbot/config-template.py

BACKEND = "Text"  # Errbot will start in text mode (console only mode) and will answer commands from there.
BOT_ASYNC = True

root_dir = os.path.abspath(os.path.dirname(__file__))

BOT_DATA_DIR = f"{root_dir}/data"
BOT_EXTRA_PLUGIN_DIR = f"{root_dir}/plugins"
BOT_EXTRA_BACKEND_DIR = f"{root_dir}/backend-plugins"

BOT_LOG_FILE = f"{root_dir}/data/errbot.log"
BOT_LOG_LEVEL = logging.DEBUG

BOT_ADMINS = (
    "@CHANGE_ME",
)  # Don't leave this as "@CHANGE_ME" if you connect your errbot to a chat system!!

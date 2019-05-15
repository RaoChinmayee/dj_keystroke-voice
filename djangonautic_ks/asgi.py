import os
import channels.asgi


os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "djangonautic_ks.settings"
)

channel_layer = channels.asgi.get_channel_layer()
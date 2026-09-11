from flask import Blueprint

api_bp = Blueprint("api", __name__, url_prefix="/api")

from . import videos, inference, live, reports, settings

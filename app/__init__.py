"""Application factory for CrossDoc Intelligence."""
from flask import Flask, jsonify, request

from config import Config


def create_app(config_object=None):
    app = Flask(__name__)
    app.config.from_object(config_object or Config)

    from .routes import bp

    app.register_blueprint(bp)
    _register_error_handlers(app)
    return app


def _register_error_handlers(app):
    """Return JSON for API routes and let pages fall back to HTML."""

    def wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        message = (
            f"The upload is too large. The total request must stay under "
            f"{limit_mb} MB."
        )
        if wants_json():
            return jsonify(error=message), 413
        return message, 413

    @app.errorhandler(404)
    def not_found(_error):
        if wants_json():
            return jsonify(error="Not found."), 404
        return "Not found.", 404

    @app.errorhandler(500)
    def server_error(error):
        app.logger.exception("Unhandled server error: %s", error)
        message = "An unexpected server error occurred."
        if wants_json():
            return jsonify(error=message), 500
        return message, 500

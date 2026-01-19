import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template  # UPDATED: add render_template
from app.controllers.fault_controller import fault_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(fault_bp)
    return app

app = create_app()



if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask
from app.controllers.fault_controller import fault_bp

app = Flask(__name__)
app.register_blueprint(fault_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
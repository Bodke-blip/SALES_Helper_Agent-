from dotenv import load_dotenv

from backend.api import create_app


load_dotenv()

app = create_app()

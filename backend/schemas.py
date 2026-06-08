from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    use_gemini_llm: bool = True
    use_local_llm: bool = True
    verbose: bool = True

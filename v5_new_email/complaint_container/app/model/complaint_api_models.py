from pydantic import BaseModel

class ComplaintAPIRequest(BaseModel):
    email: str
    snapshot_id: str
    thread_id: str
    time: int
    gcid: str

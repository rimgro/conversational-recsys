"""Version 1 vector API; shared by the query client and CPU server."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

Identifier = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
VectorValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id: Identifier
    release_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    vector: Annotated[list[VectorValue], Field(min_length=256, max_length=256)]
    k: Annotated[StrictInt, Field(ge=1, le=200)] = 200
    mode: Literal["exact", "hnsw"] = "hnsw"
    history: Annotated[list[Identifier], Field(max_length=100000)] = Field(default_factory=list)
    discovery: StrictBool = False
    query_type: Annotated[str, Field(strict=True, min_length=1, max_length=64)] = "exact"
    exclude_m4a_id: Annotated[str, Field(strict=True, max_length=128)] = ""
    exclude_artist: Annotated[str, Field(strict=True, max_length=512)] = ""
    exclude_ids: Annotated[list[Identifier], Field(max_length=100000)] = Field(default_factory=list)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    m4a_id: Identifier
    score: Annotated[VectorValue, Field(ge=-1.001, le=1.001)]


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id: Identifier
    release_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    mode: Literal["exact", "hnsw"]
    returned: Annotated[StrictInt, Field(ge=0, le=200)]
    candidates: Annotated[list[Candidate], Field(max_length=200)]
    ids: Annotated[list[Identifier], Field(max_length=200)]
    scores: Annotated[list[VectorValue], Field(max_length=200)]
    elapsed_ms: Annotated[VectorValue, Field(ge=0)]

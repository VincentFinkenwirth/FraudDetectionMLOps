"""Request and response formats of the REST API (Pydantic validates them).

I wrote the 29 features out by hand so they show up with a description in the
automatic OpenAPI docs (/docs). A unit test checks that this list is exactly
the FEATURE_COLUMNS list from preprocessing.py.

Missing values: send JSON null. For the columns where the original data uses
-1 as "missing", sending -1 is also fine; preprocessing.py handles both.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Category = str | None
Number = float | None


class ApplicationFeatures(BaseModel):
    # extra="forbid": a typo in a field name gives an error instead of a silent null.
    # allow_inf_nan=False: infinity or NaN are not valid measurements.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    income: Number = Field(description="Annual income in quantiles (0.1-0.9).")
    name_email_similarity: Number = Field(description="Similarity between e-mail and name (0-1).")
    prev_address_months_count: Number = Field(description="Months at previous address, -1 = missing.")
    current_address_months_count: Number = Field(description="Months at current address, -1 = missing.")
    customer_age: Number = Field(description="Age in years, rounded to the decade.")
    days_since_request: Number = Field(description="Days since the application was started.")
    intended_balcon_amount: Number = Field(description="Initial transferred amount; negative = missing.")
    payment_type: Category = Field(max_length=32, description="Credit payment plan (anonymised code).")
    zip_count_4w: Number = Field(description="Applications with the same zip code in the last 4 weeks.")
    velocity_6h: Number = Field(description="Average applications per hour in the last 6 hours.")
    velocity_24h: Number = Field(description="Average applications per hour in the last 24 hours.")
    velocity_4w: Number = Field(description="Average applications per hour in the last 4 weeks.")
    bank_branch_count_8w: Number = Field(description="Applications at the same branch in 8 weeks.")
    date_of_birth_distinct_emails_4w: Number = Field(description="E-mails with the same birth date in 4 weeks.")
    employment_status: Category = Field(max_length=32, description="Employment status (anonymised code).")
    credit_risk_score: Number = Field(description="Internal risk score.")
    email_is_free: Number = Field(description="1 if the e-mail domain is a free provider.")
    housing_status: Category = Field(max_length=32, description="Residential status (anonymised code).")
    phone_home_valid: Number = Field(description="1 if the home phone is valid.")
    phone_mobile_valid: Number = Field(description="1 if the mobile phone is valid.")
    bank_months_count: Number = Field(description="Age of the previous account in months, -1 = missing.")
    has_other_cards: Number = Field(description="1 if the applicant has other cards.")
    proposed_credit_limit: Number = Field(description="Proposed credit limit.")
    foreign_request: Number = Field(description="1 if the request came from another country.")
    source: Category = Field(max_length=32, description="INTERNET or TELEAPP.")
    session_length_in_minutes: Number = Field(description="Length of the web session, -1 = missing.")
    device_os: Category = Field(max_length=32, description="Operating system of the device.")
    keep_alive_session: Number = Field(description="1 if the user kept the session alive.")
    device_distinct_emails_8w: Number = Field(description="E-mails used on the device in 8 weeks, -1 = missing.")


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    event_time: datetime | None = Field(default=None, description="When the application was submitted (default: now).")
    schema_version: Literal["1.0"] = "1.0"
    features: ApplicationFeatures


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The limit keeps one request small; the simulator sends 500 per request.
    applications: list[Application] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    prediction_id: str
    application_id: str
    fraud_probability: float
    flagged_for_review: bool
    threshold: float
    model_version: str
    scored_at: datetime
    duplicate: bool = Field(description="True if this exact request was already scored earlier (retry).")


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str = Field(min_length=1, max_length=64)
    fraud_bool: Literal[0, 1]
    label_available_at: datetime | None = Field(default=None, description="When the outcome became known (default: now).")
    label_source: str | None = Field(default=None, max_length=64, description="e.g. review or audit")
    weight: float = Field(default=1.0, ge=1.0, le=10_000,
                          description="How many applications this label stands for (1 / audit share for audits).")


class LabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[Label] = Field(min_length=1, max_length=5000)


class LabelResponse(BaseModel):
    stored: int
    matched_predictions: int

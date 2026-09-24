CREATE TABLE IF NOT EXISTS products (
    id BIGSERIAL PRIMARY KEY,
    product_name TEXT NOT NULL,
    description TEXT,
    brand TEXT,
    price NUMERIC(12, 2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    predicted_category TEXT NOT NULL,
    predicted_sub_category TEXT,
    confidence NUMERIC(6, 5) CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    "timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS predictions
    ADD COLUMN IF NOT EXISTS predicted_sub_category TEXT;

CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    accuracy NUMERIC(6, 5) CHECK (accuracy IS NULL OR accuracy BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_product_id ON predictions(product_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions("timestamp");
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_category ON predictions(predicted_category);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_sub_category ON predictions(predicted_sub_category);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at);

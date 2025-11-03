CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    metadata JSONB,
    final_answer TEXT,
    reward JSONB,
    reward_stats JSONB,
    reward_audit JSONB,
    student_learning TEXT,
    teacher_learning TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',
    review_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS sessions_learning_key_idx
    ON sessions ((metadata ->> 'learning_key'));

-- Performance indexes for training data queries
CREATE INDEX IF NOT EXISTS sessions_reward_score_idx
    ON sessions(((reward_stats->>'score')::float))
    WHERE reward_stats IS NOT NULL;

CREATE INDEX IF NOT EXISTS sessions_created_at_idx
    ON sessions(created_at DESC);

CREATE INDEX IF NOT EXISTS sessions_metadata_gin_idx
    ON sessions USING gin(metadata);

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'pending';

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS review_notes TEXT;

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS reward_stats JSONB;

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS reward_audit JSONB;

CREATE TABLE IF NOT EXISTS plans (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    plan JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS step_results (
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    step_id INTEGER NOT NULL,
    trace TEXT,
    output TEXT,
    evaluation JSONB,
    attempts INTEGER,
    metadata JSONB,
    PRIMARY KEY (session_id, step_id)
);

CREATE TABLE IF NOT EXISTS step_attempts (
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    step_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    evaluation JSONB,
    PRIMARY KEY (session_id, step_id, attempt)
);

CREATE TABLE IF NOT EXISTS guidance_notes (
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    step_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    note TEXT NOT NULL,
    PRIMARY KEY (session_id, step_id, sequence)
);

CREATE TABLE IF NOT EXISTS trajectory_events (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    event JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS discovery_runs (
    id SERIAL PRIMARY KEY,
    project_root TEXT NOT NULL,
    task TEXT,
    source TEXT NOT NULL DEFAULT 'discovery',
    payload JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_registry (
    learning_key TEXT PRIMARY KEY,
    student_learning TEXT,
    teacher_learning TEXT,
    metadata JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

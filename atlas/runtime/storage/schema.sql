CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    metadata JSONB,
    final_answer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

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

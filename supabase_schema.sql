-- Supabase SQL Schema for AI Agent System
-- Run this in your Supabase SQL editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Core request tracking
CREATE TABLE IF NOT EXISTS requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    organization_id VARCHAR(255),
    problem_description TEXT NOT NULL,
    desired_outcome TEXT NOT NULL,
    current_tools JSONB DEFAULT '[]',
    estimated_hours_saved INTEGER,
    priority INTEGER DEFAULT 3,
    status VARCHAR(50) DEFAULT 'submitted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Every agent action logged for learning
CREATE TABLE IF NOT EXISTS agent_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID REFERENCES requests(id),
    trace_id VARCHAR(100),
    agent_name VARCHAR(100) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    action_details JSONB,
    result JSONB,
    success BOOLEAN DEFAULT false,
    error_message TEXT,
    tokens_used INTEGER DEFAULT 0,
    model_used VARCHAR(50),
    duration_ms INTEGER,
    learned_pattern TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent capability evolution
CREATE TABLE IF NOT EXISTS agent_improvements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    capability_before TEXT,
    capability_after TEXT,
    trigger_pattern TEXT,
    success_rate_before DECIMAL(5,2),
    success_rate_after DECIMAL(5,2),
    training_data JSONB,
    implemented_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    implemented_by VARCHAR(100) DEFAULT 'fixer'
);

-- Pattern library for reuse
CREATE TABLE IF NOT EXISTS solution_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_hash VARCHAR(64) UNIQUE,
    problem_type VARCHAR(200),
    solution_template JSONB NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    average_duration_ms INTEGER,
    applicable_agents JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent performance metrics
CREATE TABLE IF NOT EXISTS agent_metrics (
    agent_name VARCHAR(100),
    metric_date DATE,
    tasks_attempted INTEGER DEFAULT 0,
    tasks_succeeded INTEGER DEFAULT 0,
    average_duration_ms INTEGER,
    tokens_consumed INTEGER DEFAULT 0,
    patterns_learned INTEGER DEFAULT 0,
    error_types JSONB DEFAULT '{}',
    PRIMARY KEY (agent_name, metric_date)
);

-- Request feedback for learning
CREATE TABLE IF NOT EXISTS request_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID REFERENCES requests(id),
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),
    time_saved_hours INTEGER,
    would_recommend BOOLEAN,
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_requests_user ON requests(user_id);
CREATE INDEX idx_requests_status ON requests(status);
CREATE INDEX idx_interactions_request ON agent_interactions(request_id);
CREATE INDEX idx_interactions_agent ON agent_interactions(agent_name, timestamp);
CREATE INDEX idx_patterns_hash ON solution_patterns(pattern_hash);
CREATE INDEX idx_metrics_agent_date ON agent_metrics(agent_name, metric_date DESC);

-- Enable Row Level Security
ALTER TABLE requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_improvements ENABLE ROW LEVEL SECURITY;
ALTER TABLE solution_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE request_feedback ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your auth strategy)
-- For now, we'll create permissive policies
CREATE POLICY "Enable all for authenticated users" ON requests FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON agent_interactions FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON agent_improvements FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON solution_patterns FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON agent_metrics FOR ALL USING (true);
CREATE POLICY "Enable all for authenticated users" ON request_feedback FOR ALL USING (true);

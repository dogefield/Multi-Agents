# Request Portal & Database Schema
# This is the entry point where users describe problems and the foundation for all data storage

import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import uuid
import asyncio
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Database Schema
DATABASE_SCHEMA = """
-- Core request tracking
CREATE TABLE IF NOT EXISTS requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    organization_id VARCHAR(255),
    problem_description TEXT NOT NULL,
    desired_outcome TEXT NOT NULL,
    current_tools JSON DEFAULT '[]',
    estimated_hours_saved INTEGER,
    priority INTEGER DEFAULT 3,
    status VARCHAR(50) DEFAULT 'submitted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Every agent action logged for learning
CREATE TABLE IF NOT EXISTS agent_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID REFERENCES requests(id),
    trace_id VARCHAR(100),
    agent_name VARCHAR(100) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    action_details JSON,
    result JSON,
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
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(100) NOT NULL,
    capability_before TEXT,
    capability_after TEXT,
    trigger_pattern TEXT,
    success_rate_before DECIMAL(5,2),
    success_rate_after DECIMAL(5,2),
    training_data JSON,
    implemented_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    implemented_by VARCHAR(100) DEFAULT 'fixer'
);

-- Pattern library for reuse
CREATE TABLE IF NOT EXISTS solution_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_hash VARCHAR(64) UNIQUE,
    problem_type VARCHAR(200),
    solution_template JSON NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    average_duration_ms INTEGER,
    applicable_agents JSON DEFAULT '[]',
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
    error_types JSON DEFAULT '{}',
    PRIMARY KEY (agent_name, metric_date)
);

-- Request feedback for learning
CREATE TABLE IF NOT EXISTS request_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
"""

# FastAPI Request Portal
app = FastAPI(title="AI Agent Request Portal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ProblemRequest(BaseModel):
    problem_description: str
    desired_outcome: str
    current_tools: Optional[List[str]] = []
    estimated_hours_weekly: Optional[int] = None
    examples: Optional[List[str]] = []

class RequestResponse(BaseModel):
    request_id: str
    status: str
    estimated_completion: str
    assigned_agents: List[str]

# Database connection pool
class Database:
    def __init__(self):
        self.pool = None
        
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            os.getenv('DATABASE_URL', 'postgresql://localhost/agent_system'),
            min_size=10,
            max_size=20
        )
        # Initialize schema
        async with self.pool.acquire() as conn:
            await conn.execute(DATABASE_SCHEMA)
    
    async def close(self):
        await self.pool.close()

db = Database()

# API Endpoints
@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.close()

@app.post("/submit_request", response_model=RequestResponse)
async def submit_request(request: ProblemRequest, user_id: str):
    """
    Entry point for new problems. Captures natural language descriptions
    and queues them for agent processing.
    """
    async with db.pool.acquire() as conn:
        # Create request record
        request_id = await conn.fetchval("""
            INSERT INTO requests (
                user_id, problem_description, desired_outcome,
                current_tools, estimated_hours_saved
            ) VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, user_id, request.problem_description, request.desired_outcome,
            json.dumps(request.current_tools), request.estimated_hours_weekly)
        
        # Analyze request to determine initial agents
        assigned_agents = await analyze_and_assign_agents(request)
        
        # Queue for processing
        await queue_for_processing(str(request_id), assigned_agents)
        
        return RequestResponse(
            request_id=str(request_id),
            status="queued",
            estimated_completion="2-4 hours",
            assigned_agents=assigned_agents
        )

@app.get("/request_status/{request_id}")
async def get_request_status(request_id: str):
    """Check status and progress of a request"""
    async with db.pool.acquire() as conn:
        request = await conn.fetchrow(
            "SELECT * FROM requests WHERE id = $1",
            uuid.UUID(request_id)
        )
        
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")
        
        # Get recent interactions
        interactions = await conn.fetch("""
            SELECT agent_name, action_type, success, timestamp
            FROM agent_interactions 
            WHERE request_id = $1
            ORDER BY timestamp DESC
            LIMIT 10
        """, uuid.UUID(request_id))
        
        return {
            "request": dict(request),
            "recent_activity": [dict(i) for i in interactions],
            "progress_percentage": calculate_progress(interactions)
        }

@app.post("/provide_feedback/{request_id}")
async def provide_feedback(
    request_id: str,
    satisfaction: int,
    time_saved: int,
    feedback: str
):
    """Capture user feedback for continuous improvement"""
    async with db.pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO request_feedback
            (request_id, satisfaction_score, time_saved_hours, feedback_text)
            VALUES ($1, $2, $3, $4)
        """, uuid.UUID(request_id), satisfaction, time_saved, feedback)
        
        # Trigger learning from feedback
        await trigger_learning_from_feedback(request_id, satisfaction)
        
        return {"status": "feedback_recorded"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check database connection
        async with db.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "portal"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Helper Functions
async def analyze_and_assign_agents(request: ProblemRequest) -> List[str]:
    """
    Analyzes request and determines which agents should handle it.
    This is a simplified version - in production, this would use
    Claude to understand the request.
    """
    agents = []
    
    # Simple keyword matching for MVP
    problem_lower = request.problem_description.lower()
    
    if any(word in problem_lower for word in ['invoice', 'payment', 'accounting']):
        agents.append('quickbooks-agent')
        
    if any(word in problem_lower for word in ['report', 'analyze', 'data']):
        agents.append('analysis-agent')
        
    if any(word in problem_lower for word in ['email', 'notify', 'send']):
        agents.append('communication-agent')
        
    if any(word in problem_lower for word in ['website', 'ui', 'design']):
        agents.append('ui-agent')
    
    # Always include fixer for monitoring
    agents.append('fixer-agent')
    
    return agents or ['general-agent', 'fixer-agent']

async def queue_for_processing(request_id: str, agents: List[str]):
    """Queue request for agent processing via NATS"""
    # This would publish to NATS in production
    # For now, we'll just log it
    print(f"Queued request {request_id} for agents: {agents}")

def calculate_progress(interactions) -> int:
    """Estimate completion percentage based on interactions"""
    if not interactions:
        return 0
    
    success_count = sum(1 for i in interactions if i['success'])
    total_expected_steps = 5  # Simplified estimate
    
    return min(int((success_count / total_expected_steps) * 100), 95)

async def trigger_learning_from_feedback(request_id: str, satisfaction: int):
    """Queue learning job based on feedback"""
    # In production, this triggers the Fixer to analyze what worked/didn't
    print(f"Learning triggered for request {request_id} with satisfaction {satisfaction}")

# HTML Interface (Single Page App)
@app.get("/")
async def serve_portal():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Request Portal</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #4CAF50;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        .examples {
            background: #f8f8f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .examples h3 {
            margin-top: 0;
            color: #666;
            font-size: 14px;
        }
        .examples li {
            color: #666;
            margin-bottom: 8px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .tools-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .tool-chip {
            padding: 8px 12px;
            background: #e3f2fd;
            border-radius: 20px;
            font-size: 14px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tool-chip.selected {
            background: #2196F3;
            color: white;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        .status.success {
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #4caf50;
        }
        .status.error {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tell us what's frustrating you</h1>
        <p class="subtitle">Describe your problem in plain English, and our AI agents will figure out how to help.</p>
        
        <div class="examples">
            <h3>EXAMPLES:</h3>
            <ul>
                <li>"I spend 3 hours every Monday reconciling receipts"</li>
                <li>"Customers complain they don't know their order status"</li>
                <li>"I never know which products are actually making money"</li>
                <li>"Creating monthly reports takes forever and they're always late"</li>
            </ul>
        </div>
        
        <form id="requestForm">
            <div class="form-group">
                <label for="problem">What problem are you trying to solve?</label>
                <textarea 
                    id="problem" 
                    name="problem"
                    placeholder="Describe what's taking too much time or causing frustration..."
                    required
                ></textarea>
            </div>
            
            <div class="form-group">
                <label for="outcome">What would success look like?</label>
                <textarea 
                    id="outcome" 
                    name="outcome"
                    placeholder="Describe your ideal outcome or how you'd like this to work..."
                    required
                ></textarea>
            </div>
            
            <div class="form-group">
                <label>What tools do you currently use? (optional)</label>
                <div class="tools-grid">
                    <div class="tool-chip" data-tool="quickbooks">QuickBooks</div>
                    <div class="tool-chip" data-tool="stripe">Stripe</div>
                    <div class="tool-chip" data-tool="shopify">Shopify</div>
                    <div class="tool-chip" data-tool="gmail">Gmail</div>
                    <div class="tool-chip" data-tool="slack">Slack</div>
                    <div class="tool-chip" data-tool="excel">Excel</div>
                    <div class="tool-chip" data-tool="salesforce">Salesforce</div>
                    <div class="tool-chip" data-tool="notion">Notion</div>
                </div>
            </div>
            
            <div class="form-group">
                <label for="hours">How many hours per week does this take? (optional)</label>
                <input 
                    type="number" 
                    id="hours" 
                    name="hours" 
                    min="0" 
                    max="168"
                    style="width: 100px; padding: 8px; border: 2px solid #e0e0e0; border-radius: 8px;"
                >
            </div>
            
            <button type="submit" id="submitBtn">Submit Request</button>
            
            <div id="status" class="status"></div>
        </form>
    </div>
    
    <script>
        // Tool selection
        const tools = [];
        document.querySelectorAll('.tool-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                chip.classList.toggle('selected');
                const tool = chip.dataset.tool;
                const index = tools.indexOf(tool);
                if (index > -1) {
                    tools.splice(index, 1);
                } else {
                    tools.push(tool);
                }
            });
        });
        
        // Form submission
        document.getElementById('requestForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const status = document.getElementById('status');
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
            
            const formData = {
                problem_description: document.getElementById('problem').value,
                desired_outcome: document.getElementById('outcome').value,
                current_tools: tools,
                estimated_hours_weekly: parseInt(document.getElementById('hours').value) || null
            };
            
            try {
                const response = await fetch('/submit_request?user_id=demo-user', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                status.className = 'status success';
                status.innerHTML = `
                    <strong>Success!</strong> Your request has been submitted.<br>
                    Request ID: ${result.request_id}<br>
                    Assigned agents: ${result.assigned_agents.join(', ')}<br>
                    Estimated completion: ${result.estimated_completion}
                `;
                status.style.display = 'block';
                
                // Reset form
                document.getElementById('requestForm').reset();
                document.querySelectorAll('.tool-chip.selected').forEach(chip => {
                    chip.classList.remove('selected');
                });
                tools.length = 0;
                
            } catch (error) {
                status.className = 'status error';
                status.innerHTML = `<strong>Error:</strong> ${error.message}`;
                status.style.display = 'block';
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit Request';
            }
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

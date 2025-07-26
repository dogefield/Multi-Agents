# Portal with Supabase Integration
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_ANON_KEY"))

if not supabase_url or not supabase_key:
    logger.error("Supabase credentials not found in environment variables!")
    logger.info("Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file")
    supabase: Optional[Client] = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info(f"✅ Connected to Supabase: {supabase_url}")

# FastAPI app
app = FastAPI(title="AI Agent Portal - Supabase Edition")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ProblemRequest(BaseModel):
    problem_description: str
    desired_outcome: str
    current_tools: Optional[List[str]] = []
    estimated_hours_weekly: Optional[int] = None

class RequestResponse(BaseModel):
    request_id: str
    status: str
    estimated_completion: str
    assigned_agents: List[str]

@app.post("/submit_request", response_model=RequestResponse)
async def submit_request(request: ProblemRequest, user_id: str = "demo-user"):
    """Submit a new problem request"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Create request in Supabase
        data = {
            "user_id": user_id,
            "problem_description": request.problem_description,
            "desired_outcome": request.desired_outcome,
            "current_tools": request.current_tools,
            "estimated_hours_saved": request.estimated_hours_weekly,
            "status": "submitted"
        }
        
        result = supabase.table("requests").insert(data).execute()
        request_data = result.data[0]
        request_id = request_data["id"]
        
        # Analyze and assign agents
        assigned_agents = analyze_request(request)
        
        # Log initial interaction
        for agent in assigned_agents:
            supabase.table("agent_interactions").insert({
                "request_id": request_id,
                "agent_name": agent,
                "action_type": "assigned",
                "action_details": {"reason": "matched_keywords"},
                "success": True
            }).execute()
        
        logger.info(f"✅ Created request {request_id} with agents: {assigned_agents}")
        
        return RequestResponse(
            request_id=request_id,
            status="submitted",
            estimated_completion="2-4 hours",
            assigned_agents=assigned_agents
        )
        
    except Exception as e:
        logger.error(f"Error creating request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/request_status/{request_id}")
async def get_request_status(request_id: str):
    """Check status and progress of a request"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Get request
        request_result = supabase.table("requests").select("*").eq("id", request_id).execute()
        
        if not request_result.data:
            raise HTTPException(status_code=404, detail="Request not found")
        
        request_data = request_result.data[0]
        
        # Get recent interactions
        interactions_result = supabase.table("agent_interactions")\
            .select("agent_name, action_type, success, timestamp")\
            .eq("request_id", request_id)\
            .order("timestamp", desc=True)\
            .limit(10)\
            .execute()
        
        interactions = interactions_result.data
        
        return {
            "request": request_data,
            "recent_activity": interactions,
            "progress_percentage": calculate_progress(interactions)
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/metrics")
async def get_agent_metrics():
    """Get agent performance metrics"""
    if not supabase:
        return {"error": "Supabase not configured"}
    
    try:
        # Get today's metrics
        today = datetime.now().date()
        metrics_result = supabase.table("agent_metrics")\
            .select("*")\
            .eq("metric_date", str(today))\
            .execute()
        
        return {"metrics": metrics_result.data, "date": str(today)}
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": str(e)}

def analyze_request(request: ProblemRequest) -> List[str]:
    """Analyze request and assign appropriate agents"""
    agents = []
    problem_lower = request.problem_description.lower()
    
    # Keywords for each agent type
    agent_keywords = {
        "quickbooks-agent": ["invoice", "payment", "accounting", "receipt", "reconcile", "bookkeeping"],
        "analysis-agent": ["report", "analyze", "data", "insight", "trend", "metric", "dashboard"],
        "communication-agent": ["email", "notify", "send", "alert", "message", "notification"],
        "ui-agent": ["website", "interface", "design", "user experience", "frontend"],
        "strategy-agent": ["plan", "strategy", "roadmap", "growth", "optimization"]
    }
    
    # Check keywords
    for agent, keywords in agent_keywords.items():
        if any(keyword in problem_lower for keyword in keywords):
            agents.append(agent)
    
    # Always include fixer for monitoring
    agents.append("fixer-agent")
    
    # Add general agent if no specific match
    if len(agents) == 1:  # Only fixer
        agents.insert(0, "general-agent")
    
    return agents

def calculate_progress(interactions) -> int:
    """Calculate progress based on interactions"""
    if not interactions:
        return 0
    
    # Count successful actions
    success_count = sum(1 for i in interactions if i.get('success'))
    total_count = len(interactions)
    
    if total_count == 0:
        return 0
    
    # Calculate percentage (max 95% until marked complete)
    progress = min(int((success_count / total_count) * 100), 95)
    
    # Check if any interaction is completion
    if any(i.get('action_type') == 'completed' for i in interactions):
        progress = 100
    
    return progress

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    supabase_status = "connected" if supabase else "not configured"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "supabase": supabase_status
    }

@app.get("/setup")
async def setup_instructions():
    """Show setup instructions"""
    return HTMLResponse("""
    <html>
    <head>
        <title>Supabase Setup Instructions</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   max-width: 800px; margin: 0 auto; padding: 40px 20px; }
            .step { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
            code { background: #e0e0e0; padding: 2px 6px; border-radius: 4px; }
            pre { background: #2d2d2d; color: #fff; padding: 15px; border-radius: 8px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🚀 Supabase Setup for AI Agent System</h1>
        
        <div class="step">
            <h2>Step 1: Create Supabase Project</h2>
            <p>Go to <a href="https://app.supabase.com" target="_blank">app.supabase.com</a> and create a new project (free tier is fine)</p>
        </div>
        
        <div class="step">
            <h2>Step 2: Run the Database Schema</h2>
            <p>1. In Supabase dashboard, go to SQL Editor</p>
            <p>2. Create new query</p>
            <p>3. Copy contents of <code>supabase_schema.sql</code> and run it</p>
        </div>
        
        <div class="step">
            <h2>Step 3: Get Your Credentials</h2>
            <p>In Supabase dashboard Settings → API:</p>
            <ul>
                <li>Project URL → <code>SUPABASE_URL</code></li>
                <li>anon/public key → <code>SUPABASE_ANON_KEY</code></li>
                <li>service_role key → <code>SUPABASE_SERVICE_KEY</code></li>
            </ul>
        </div>
        
        <div class="step">
            <h2>Step 4: Update .env File</h2>
            <pre>SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
ANTHROPIC_API_KEY=your-anthropic-key</pre>
        </div>
        
        <div class="step">
            <h2>Step 5: Restart the Portal</h2>
            <p>The portal will now use Supabase instead of local PostgreSQL!</p>
        </div>
        
        <p><a href="/">← Back to Portal</a></p>
    </body>
    </html>
    """)

@app.get("/", response_class=HTMLResponse)
async def serve_portal():
    """Serve the main portal interface"""
    supabase_configured = supabase is not None
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Request Portal</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
        }}
        textarea {{
            width: 100%;
            min-height: 120px;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }}
        textarea:focus {{
            outline: none;
            border-color: #4CAF50;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }}
        button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #45a049;
        }}
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        .status {{
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }}
        .status.success {{
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #4caf50;
        }}
        .status.error {{
            background: #ffebee;
            color: #c62828;
            border: 1px solid #f44336;
        }}
        .examples {{
            background: #f8f8f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .examples h3 {{
            margin-top: 0;
            color: #666;
            font-size: 14px;
        }}
        .examples li {{
            color: #666;
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Agent Request Portal</h1>
        <p class="subtitle">Describe your problem in plain English, and our AI agents will figure out how to help.</p>
        
        {"" if supabase_configured else '''
        <div class="warning">
            <strong>⚠️ Supabase not configured!</strong><br>
            The portal is running but can't save data. <a href="/setup">See setup instructions</a>
        </div>
        '''}
        
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
            
            <button type="submit" id="submitBtn" {"" if supabase_configured else 'disabled'}>
                {"Submit Request" if supabase_configured else "Configure Supabase First"}
            </button>
            
            <div id="status" class="status"></div>
        </form>
    </div>
    
    <script>
        document.getElementById('requestForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const status = document.getElementById('status');
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
            
            const formData = {{
                problem_description: document.getElementById('problem').value,
                desired_outcome: document.getElementById('outcome').value,
                current_tools: []
            }};
            
            try {{
                const response = await fetch('/submit_request?user_id=demo-user', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(formData)
                }});
                
                const result = await response.json();
                
                if (response.ok) {{
                    status.className = 'status success';
                    status.innerHTML = `
                        <strong>Success!</strong> Your request has been submitted.<br>
                        Request ID: ${{result.request_id}}<br>
                        Assigned agents: ${{result.assigned_agents.join(', ')}}<br>
                        Estimated completion: ${{result.estimated_completion}}
                    `;
                    status.style.display = 'block';
                    
                    // Reset form
                    document.getElementById('requestForm').reset();
                }} else {{
                    throw new Error(result.detail || 'Submission failed');
                }}
                
            }} catch (error) {{
                status.className = 'status error';
                status.innerHTML = `<strong>Error:</strong> ${{error.message}}`;
                status.style.display = 'block';
            }} finally {{
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit Request';
            }}
        }});
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Agent Portal with Supabase")
    print(f"📊 Supabase Status: {'✅ Connected' if supabase else '❌ Not configured'}")
    
    if not supabase:
        print("\n⚠️  To configure Supabase:")
        print("1. Create account at https://app.supabase.com")
        print("2. Update .env with your credentials")
        print("3. Run schema from supabase_schema.sql")
        print("\n📖 Visit http://localhost:8000/setup for detailed instructions")
    
    print("\n🌐 Portal running at http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
